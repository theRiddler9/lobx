// pybind11 bindings: expose the C++ OrderBook to Python as lobx._native.NativeBook.
//
// The binding speaks only plain Python ints and tuples. Building the Order/Fill dataclasses is
// done by the Python wrapper in src/lobx/engine/native.py, which keeps this file small and keeps
// the Python contract types defined in exactly one place (src/lobx/types.py).
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <optional>
#include <tuple>
#include <vector>

#include "lobx/order_book.hpp"

namespace py = pybind11;

namespace {

// One fill as (maker_id, taker_id, taker_side, price, qty, ts); side is 0 for buy, 1 for sell.
using FillTuple =
    std::tuple<std::int64_t, std::int64_t, int, std::int64_t, std::int64_t, std::int64_t>;

// Raise the Python exception classes from lobx.errors, so callers catch the same types whichever
// engine they use. The import happens inside the translator, i.e. only when an error occurs.
void translate_error(std::exception_ptr error) {
  try {
    if (error) std::rethrow_exception(error);
  } catch (
      const lobx::DuplicateOrderIdError& e) {  // subclass first: it is also an InvalidOrderError
    py::object cls = py::module_::import("lobx.errors").attr("DuplicateOrderIdError");
    PyErr_SetString(cls.ptr(), e.what());
  } catch (const lobx::InvalidOrderError& e) {
    py::object cls = py::module_::import("lobx.errors").attr("InvalidOrderError");
    PyErr_SetString(cls.ptr(), e.what());
  }
}

}  // namespace

PYBIND11_MODULE(_native, m) {
  m.doc() = "C++ matching engine (see cpp/include/lobx/order_book.hpp)";
  py::register_exception_translator(translate_error);

  py::class_<lobx::OrderBook>(m, "NativeBook")
      .def(py::init<>())
      .def(
          "submit",
          [](lobx::OrderBook& book, std::int64_t id, std::int64_t agent, int side, int type,
             std::optional<std::int64_t> price, std::int64_t qty, std::int64_t ts) {
            const lobx::Order order{id,
                                    agent,
                                    side == 0 ? lobx::Side::Buy : lobx::Side::Sell,
                                    type == 0 ? lobx::OrderType::Limit : lobx::OrderType::Market,
                                    price,
                                    qty,
                                    ts};
            const std::vector<lobx::Fill> fills = book.submit(order);
            std::vector<FillTuple> out;
            out.reserve(fills.size());
            for (const lobx::Fill& f : fills) {
              out.emplace_back(f.maker_id, f.taker_id, f.taker_side == lobx::Side::Buy ? 0 : 1,
                               f.price, f.qty, f.ts);
            }
            return out;
          },
          py::arg("id"), py::arg("agent"), py::arg("side"), py::arg("type"), py::arg("price"),
          py::arg("qty"), py::arg("ts"))
      .def("cancel", &lobx::OrderBook::cancel, py::arg("order_id"), py::arg("ts"))
      .def(
          "snapshot",
          [](const lobx::OrderBook& book, std::size_t depth) {
            const lobx::Snapshot s = book.snapshot(depth);
            return std::make_tuple(s.ts, s.bids, s.asks);
          },
          py::arg("depth"))
      .def("best_bid", &lobx::OrderBook::best_bid)
      .def("best_ask", &lobx::OrderBook::best_ask)
      .def("resting_orders", &lobx::OrderBook::resting_orders);
}
