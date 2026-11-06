// Native cable solver for the T4 cables (Node-API addon; build with build-native.sh).
// CablePool keeps persistent threads, each owning a contiguous range of cells. Commands:
//   solve            one implicit (backward-Euler) Hines step of every cable: the same arithmetic in the same
//                    order as fe() in sim.worker.js (build without fast-math and FMA contraction to keep it so)
//   assemble_inputs  reset the synapse compartments and add the visual input conductances (step (2) of the
//                    worker's time loop)
// and assemble_lpi_inputs (T4b -> LPi15 for point-model T4b; inactive in the circuit runs).
// The typed arrays belong to native-cable-solver.mjs; the addon only keeps pointers to them.
#include <node_api.h>
#include <algorithm>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

// A pointer into a JavaScript typed array.
template <typename T> struct TypedView { T* data; size_t length; };

template <typename T>
TypedView<T> typed_view(napi_env env, napi_value value, napi_typedarray_type expected) {
  napi_typedarray_type type; size_t length = 0; void* data = nullptr;
  napi_value buffer; size_t byte_offset = 0;
  if (napi_get_typedarray_info(env, value, &type, &length, &data, &buffer, &byte_offset) != napi_ok
      || type != expected) throw std::runtime_error("Unexpected typed-array argument");
  return {static_cast<T*>(data), length};
}

// The packed cable arrays and the worker threads.
class CablePool {
 public:
  TypedView<int32_t> offsets, parents, forward_offsets, forward_order, backward_offsets, backward_order;
  TypedView<double> coupling, matrix_diagonal, capacitance, leak;
  TypedView<double> voltage, next, conductance, current, diagonal, rhs;
  TypedView<int32_t> input_touched_offsets{}, input_touched_compartments{};
  TypedView<int32_t> input_contact_offsets{}, input_contact_compartments{}, input_per_contact{};
  TypedView<double> input_units{}, input_reversals{}, input_frames{};
  TypedView<int32_t> lpi_input_offsets{}, lpi_input_compartments{};
  TypedView<double> lpi_input_release{}, lpi_conductance{}, lpi_current{}, lpi_parameters{};
  bool input_configured = false;
  bool lpi_input_configured = false;
  std::vector<std::thread> workers;
  std::mutex mutex;
  std::condition_variable work_cv, done_cv;
  uint64_t generation = 0;
  int done = 0;
  bool stopping = false;
  double dt_sec = 0;
  int command = 0;

  CablePool(
      TypedView<int32_t> offsets_, TypedView<int32_t> parents_, TypedView<double> coupling_,
      TypedView<double> matrix_diagonal_, TypedView<double> capacitance_, TypedView<double> leak_,
      TypedView<int32_t> forward_offsets_, TypedView<int32_t> forward_order_,
      TypedView<int32_t> backward_offsets_, TypedView<int32_t> backward_order_,
      TypedView<double> voltage_, TypedView<double> next_, TypedView<double> conductance_,
      TypedView<double> current_, TypedView<double> diagonal_, TypedView<double> rhs_, int threads)
      : offsets(offsets_), parents(parents_), forward_offsets(forward_offsets_),
        forward_order(forward_order_), backward_offsets(backward_offsets_), backward_order(backward_order_),
        coupling(coupling_), matrix_diagonal(matrix_diagonal_), capacitance(capacitance_), leak(leak_),
        voltage(voltage_), next(next_), conductance(conductance_), current(current_),
        diagonal(diagonal_), rhs(rhs_) {
    const int cell_count = static_cast<int>(offsets.length - 1);
    const int worker_count = std::max(1, std::min(threads, cell_count));
    int start = 0;
    for (int worker = 0; worker < worker_count; ++worker) {
      const int remaining = worker_count - worker;
      int end;
      if (remaining == 1) {
        end = cell_count;
      } else {
        const int minimum_end = start + 1;
        const int maximum_end = cell_count - (remaining - 1);
        const double target = offsets.data[start]
          + static_cast<double>(offsets.data[cell_count] - offsets.data[start]) / remaining;
        end = minimum_end;
        while (end < maximum_end && offsets.data[end] < target) ++end;
      }
      workers.emplace_back([this, start, end] { worker_loop(start, end); });
      start = end;
    }
  }

  ~CablePool() { close(); }

  // One time step of every cable; voltage and next are swapped afterwards, as in the JavaScript solver.
  void solve(double dt) {
    run(1, dt);
    std::swap(voltage.data, next.data);
  }

  void configure_inputs(
      TypedView<int32_t> touched_offsets, TypedView<int32_t> touched_compartments,
      TypedView<int32_t> contact_offsets, TypedView<int32_t> contact_compartments,
      TypedView<double> units, TypedView<double> reversals, TypedView<double> frames,
      TypedView<int32_t> per_contact) {
    input_touched_offsets = touched_offsets;
    input_touched_compartments = touched_compartments;
    input_contact_offsets = contact_offsets;
    input_contact_compartments = contact_compartments;
    input_units = units;
    input_reversals = reversals;
    input_frames = frames;
    input_per_contact = per_contact;
    input_configured = true;
  }

  void assemble_inputs() {
    if (!input_configured) throw std::runtime_error("Native input assembly is not configured");
    run(2, 0);
  }

  void configure_lpi_inputs(
      TypedView<int32_t> contact_offsets, TypedView<int32_t> contact_compartments,
      TypedView<double> release, TypedView<double> conductance,
      TypedView<double> current, TypedView<double> parameters) {
    lpi_input_offsets = contact_offsets;
    lpi_input_compartments = contact_compartments;
    lpi_input_release = release;
    lpi_conductance = conductance;
    lpi_current = current;
    lpi_parameters = parameters;
    lpi_input_configured = true;
  }

  // Single-threaded T4b -> LPi15 assembly for point-model T4b (inactive).
  void assemble_lpi_inputs() {
    if (!lpi_input_configured) throw std::runtime_error("Native LPi15 input assembly is not configured");
    std::fill(lpi_conductance.data, lpi_conductance.data + lpi_conductance.length, 0.0);
    std::fill(lpi_current.data, lpi_current.data + lpi_current.length, 0.0);
    const double per_contact_ns = lpi_parameters.data[0];
    const double reversal_mv = lpi_parameters.data[1];
    for (size_t cell = 0; cell + 1 < lpi_input_offsets.length; ++cell) {
      const int32_t start = lpi_input_offsets.data[cell];
      const int32_t end = lpi_input_offsets.data[cell + 1];
      if (start == end) continue;
      const double release = lpi_input_release.data[cell];
      if (release <= 0) continue;
      const double conductance_s = per_contact_ns * release * 1e-9;
      const double current_increment = conductance_s * reversal_mv / 1e3;
      for (int32_t contact = start; contact < end; ++contact) {
        const int32_t compartment = lpi_input_compartments.data[contact];
        if (compartment < 0 || static_cast<size_t>(compartment) >= lpi_conductance.length) continue;
        lpi_conductance.data[compartment] += conductance_s;
        lpi_current.data[compartment] += current_increment;
      }
    }
  }

  // Run a command (1 = solve, 2 = assemble inputs) on every thread and wait until all are done.
  void run(int next_command, double dt) {
    std::unique_lock<std::mutex> lock(mutex);
    dt_sec = dt;
    command = next_command;
    done = 0;
    ++generation;
    work_cv.notify_all();
    done_cv.wait(lock, [this] { return done == static_cast<int>(workers.size()); });
  }

  void close() {
    {
      std::lock_guard<std::mutex> lock(mutex);
      if (stopping) return;
      stopping = true;
      ++generation;
    }
    work_cv.notify_all();
    for (auto& worker : workers) if (worker.joinable()) worker.join();
  }

 private:
  // Thread body: wait for a new generation, run the command on this thread's cells, report done.
  void worker_loop(int cell_start, int cell_end) {
    uint64_t seen = 0;
    while (true) {
      double dt;
      int next_command;
      {
        std::unique_lock<std::mutex> lock(mutex);
        work_cv.wait(lock, [this, seen] { return stopping || generation != seen; });
        if (stopping) return;
        seen = generation;
        dt = dt_sec;
        next_command = command;
      }
      if (next_command == 2) assemble_input_range(cell_start, cell_end);
      else solve_range(cell_start, cell_end, dt);
      {
        std::lock_guard<std::mutex> lock(mutex);
        ++done;
        if (done == static_cast<int>(workers.size())) done_cv.notify_one();
      }
    }
  }

  // Implicit Euler step (Hines algorithm) of cells [cell_start, cell_end); leak reversal -65 mV.
  void solve_range(int cell_start, int cell_end, double dt) {
    for (int cell = cell_start; cell < cell_end; ++cell) {
      for (int32_t node = offsets.data[cell]; node < offsets.data[cell + 1]; ++node) {
        const double capacitance_over_dt = capacitance.data[node] / dt;
        diagonal.data[node] = matrix_diagonal.data[node] + capacitance_over_dt + conductance.data[node];
        rhs.data[node] = capacitance_over_dt * voltage.data[node]
          + leak.data[node] * (-65.0 / 1000.0) + current.data[node];
      }
      for (int32_t order = forward_offsets.data[cell]; order < forward_offsets.data[cell + 1]; ++order) {
        const int32_t node = forward_order.data[order], parent = parents.data[node];
        const double factor = coupling.data[node] / diagonal.data[node];
        diagonal.data[parent] -= coupling.data[node] * factor;
        rhs.data[parent] -= factor * rhs.data[node];
      }
      for (int32_t order = backward_offsets.data[cell]; order < backward_offsets.data[cell + 1]; ++order) {
        const int32_t node = backward_order.data[order], parent = parents.data[node];
        next.data[node] = parent < 0 ? rhs.data[node] / diagonal.data[node]
          : (rhs.data[node] - coupling.data[node] * next.data[parent]) / diagonal.data[node];
      }
    }
  }

  // Visual input conductances of cells [cell_start, cell_end): reset, then add each type at its contacts.
  void assemble_input_range(int cell_start, int cell_end) {
    const int cell_count = static_cast<int>(offsets.length - 1);
    const int kind_count = static_cast<int>(input_reversals.length);
    for (int cell = cell_start; cell < cell_end; ++cell) {
      const int32_t cable_start = offsets.data[cell];
      for (int32_t index = input_touched_offsets.data[cell];
           index < input_touched_offsets.data[cell + 1]; ++index) {
        const int32_t compartment = cable_start + input_touched_compartments.data[index];
        conductance.data[compartment] = 0;
        current.data[compartment] = 0;
      }
      for (int kind = 0; kind < kind_count; ++kind) {
        const int mapping = cell * kind_count + kind;
        const int32_t start = input_contact_offsets.data[mapping];
        const int32_t end = input_contact_offsets.data[mapping + 1];
        const int32_t count = end - start;
        if (!count) continue;
        double conductance_s = input_frames.data[kind * cell_count + cell] * input_units.data[cell];
        if (!input_per_contact.data[0]) conductance_s /= count;
        if (conductance_s <= 0) continue;
        const double current_increment = conductance_s * input_reversals.data[kind] / 1e3;
        for (int32_t contact = start; contact < end; ++contact) {
          const int32_t compartment = cable_start + input_contact_compartments.data[contact];
          conductance.data[compartment] += conductance_s;
          current.data[compartment] += current_increment;
        }
      }
    }
  }
};

// Node-API functions exported to native-cable-solver.mjs.
static CablePool* pool = nullptr;

napi_value configure(napi_env env, napi_callback_info info) {
  size_t argc = 17; napi_value argv[17];
  napi_get_cb_info(env, info, &argc, argv, nullptr, nullptr);
  if (argc != 17) { napi_throw_type_error(env, nullptr, "Native cable configure expects 17 arguments"); return nullptr; }
  try {
    int32_t threads = 1; napi_get_value_int32(env, argv[16], &threads);
    delete pool;
    pool = new CablePool(
      typed_view<int32_t>(env, argv[0], napi_int32_array),
      typed_view<int32_t>(env, argv[1], napi_int32_array),
      typed_view<double>(env, argv[2], napi_float64_array),
      typed_view<double>(env, argv[3], napi_float64_array),
      typed_view<double>(env, argv[4], napi_float64_array),
      typed_view<double>(env, argv[5], napi_float64_array),
      typed_view<int32_t>(env, argv[6], napi_int32_array),
      typed_view<int32_t>(env, argv[7], napi_int32_array),
      typed_view<int32_t>(env, argv[8], napi_int32_array),
      typed_view<int32_t>(env, argv[9], napi_int32_array),
      typed_view<double>(env, argv[10], napi_float64_array),
      typed_view<double>(env, argv[11], napi_float64_array),
      typed_view<double>(env, argv[12], napi_float64_array),
      typed_view<double>(env, argv[13], napi_float64_array),
      typed_view<double>(env, argv[14], napi_float64_array),
      typed_view<double>(env, argv[15], napi_float64_array), threads);
  } catch (const std::exception& error) { napi_throw_error(env, nullptr, error.what()); return nullptr; }
  napi_value result; napi_get_boolean(env, true, &result); return result;
}

napi_value solve(napi_env env, napi_callback_info info) {
  size_t argc = 1; napi_value argv[1]; napi_get_cb_info(env, info, &argc, argv, nullptr, nullptr);
  if (!pool || argc != 1) { napi_throw_error(env, nullptr, "Native cable pool is not configured"); return nullptr; }
  double dt = 0; napi_get_value_double(env, argv[0], &dt); pool->solve(dt);
  napi_value result; napi_get_boolean(env, true, &result); return result;
}

napi_value configure_inputs(napi_env env, napi_callback_info info) {
  size_t argc = 8; napi_value argv[8]; napi_get_cb_info(env, info, &argc, argv, nullptr, nullptr);
  if (!pool || argc != 8) { napi_throw_error(env, nullptr, "Native input configure expects 8 arguments"); return nullptr; }
  try {
    pool->configure_inputs(
      typed_view<int32_t>(env, argv[0], napi_int32_array),
      typed_view<int32_t>(env, argv[1], napi_int32_array),
      typed_view<int32_t>(env, argv[2], napi_int32_array),
      typed_view<int32_t>(env, argv[3], napi_int32_array),
      typed_view<double>(env, argv[4], napi_float64_array),
      typed_view<double>(env, argv[5], napi_float64_array),
      typed_view<double>(env, argv[6], napi_float64_array),
      typed_view<int32_t>(env, argv[7], napi_int32_array));
  } catch (const std::exception& error) { napi_throw_error(env, nullptr, error.what()); return nullptr; }
  napi_value result; napi_get_boolean(env, true, &result); return result;
}

napi_value assemble_inputs(napi_env env, napi_callback_info info) {
  if (!pool) { napi_throw_error(env, nullptr, "Native cable pool is not configured"); return nullptr; }
  try { pool->assemble_inputs(); }
  catch (const std::exception& error) { napi_throw_error(env, nullptr, error.what()); return nullptr; }
  napi_value result; napi_get_boolean(env, true, &result); return result;
}

napi_value configure_lpi_inputs(napi_env env, napi_callback_info info) {
  size_t argc = 6; napi_value argv[6]; napi_get_cb_info(env, info, &argc, argv, nullptr, nullptr);
  if (!pool || argc != 6) { napi_throw_type_error(env, nullptr, "Native LPi15 input configure expects 6 arguments"); return nullptr; }
  try {
    pool->configure_lpi_inputs(
      typed_view<int32_t>(env, argv[0], napi_int32_array),
      typed_view<int32_t>(env, argv[1], napi_int32_array),
      typed_view<double>(env, argv[2], napi_float64_array),
      typed_view<double>(env, argv[3], napi_float64_array),
      typed_view<double>(env, argv[4], napi_float64_array),
      typed_view<double>(env, argv[5], napi_float64_array));
  } catch (const std::exception& error) { napi_throw_error(env, nullptr, error.what()); return nullptr; }
  napi_value result; napi_get_boolean(env, true, &result); return result;
}

napi_value assemble_lpi_inputs(napi_env env, napi_callback_info info) {
  if (!pool) { napi_throw_error(env, nullptr, "Native cable pool is not configured"); return nullptr; }
  try { pool->assemble_lpi_inputs(); }
  catch (const std::exception& error) { napi_throw_error(env, nullptr, error.what()); return nullptr; }
  napi_value result; napi_get_boolean(env, true, &result); return result;
}

napi_value close_pool(napi_env env, napi_callback_info info) {
  delete pool; pool = nullptr; napi_value result; napi_get_undefined(env, &result); return result;
}

napi_value init(napi_env env, napi_value exports) {
  napi_value fn;
  napi_create_function(env, "configure", NAPI_AUTO_LENGTH, configure, nullptr, &fn);
  napi_set_named_property(env, exports, "configure", fn);
  napi_create_function(env, "solve", NAPI_AUTO_LENGTH, solve, nullptr, &fn);
  napi_set_named_property(env, exports, "solve", fn);
  napi_create_function(env, "configureInputs", NAPI_AUTO_LENGTH, configure_inputs, nullptr, &fn);
  napi_set_named_property(env, exports, "configureInputs", fn);
  napi_create_function(env, "assembleInputs", NAPI_AUTO_LENGTH, assemble_inputs, nullptr, &fn);
  napi_set_named_property(env, exports, "assembleInputs", fn);
  napi_create_function(env, "configureLpiInputs", NAPI_AUTO_LENGTH, configure_lpi_inputs, nullptr, &fn);
  napi_set_named_property(env, exports, "configureLpiInputs", fn);
  napi_create_function(env, "assembleLpiInputs", NAPI_AUTO_LENGTH, assemble_lpi_inputs, nullptr, &fn);
  napi_set_named_property(env, exports, "assembleLpiInputs", fn);
  napi_create_function(env, "close", NAPI_AUTO_LENGTH, close_pool, nullptr, &fn);
  napi_set_named_property(env, exports, "close", fn);
  return exports;
}

NAPI_MODULE(yifei_native_cable, init)
