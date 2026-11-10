// pybind11 bindings: expose C++ engines and asynchronous gateway to Python.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <chrono>
#include <cstdint>
#include <optional>
#include <tuple>
#include <vector>

#include "lobx/concurrent_order_book.hpp"
#include "lobx/gateway.hpp"
#include "lobx/order_book.hpp"

namespace py = pybind11;

namespace {

using FillTuple =
    std::tuple<std::int64_t, std::int64_t, int, std::int64_t, std::int64_t, std::int64_t>;
using EventTuple =
    std::tuple<uint64_t, int, std::int64_t, std::int64_t, std::int64_t, std::int64_t, std::int64_t, bool>;

void translate_error(std::exception_ptr error) {
  try {
    if (error) std::rethrow_exception(error);
  } catch (const lobx::DuplicateOrderIdError& e) {
    py::object cls = py::module_::import("lobx.errors").attr("DuplicateOrderIdError");
    PyErr_SetString(cls.ptr(), e.what());
  } catch (const lobx::InvalidOrderError& e) {
    py::object cls = py::module_::import("lobx.errors").attr("InvalidOrderError");
    PyErr_SetString(cls.ptr(), e.what());
  }
}

template <typename BookType>
std::vector<FillTuple> submit_order_helper(BookType& book, std::int64_t id, std::int64_t agent,
                                           int side, int type, std::optional<std::int64_t> price,
                                           std::int64_t qty, std::int64_t ts) {
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
    out.emplace_back(f.maker_id, f.taker_id, f.taker_side == lobx::Side::Buy ? 0 : 1, f.price,
                     f.qty, f.ts);
  }
  return out;
}

}  // namespace

PYBIND11_MODULE(_native, m) {
  m.doc() = "C++ matching engine and concurrent gateway";
  py::register_exception_translator(translate_error);

  py::class_<lobx::OrderBook>(m, "NativeBook")
      .def(py::init<>())
      .def("submit", &submit_order_helper<lobx::OrderBook>, py::arg("id"), py::arg("agent"),
           py::arg("side"), py::arg("type"), py::arg("price"), py::arg("qty"), py::arg("ts"))
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

  py::class_<lobx::ConcurrentOrderBook>(m, "NativeConcurrentBook")
      .def(py::init<>())
      .def("submit", &submit_order_helper<lobx::ConcurrentOrderBook>, py::arg("id"),
           py::arg("agent"), py::arg("side"), py::arg("type"), py::arg("price"), py::arg("qty"),
           py::arg("ts"))
      .def("cancel", &lobx::ConcurrentOrderBook::cancel, py::arg("order_id"), py::arg("ts"))
      .def(
          "snapshot",
          [](const lobx::ConcurrentOrderBook& book, std::size_t depth) {
            const lobx::Snapshot s = book.snapshot(depth);
            return std::make_tuple(s.ts, s.bids, s.asks);
          },
          py::arg("depth"))
      .def("best_bid", &lobx::ConcurrentOrderBook::best_bid)
      .def("best_ask", &lobx::ConcurrentOrderBook::best_ask)
      .def("resting_orders", &lobx::ConcurrentOrderBook::resting_orders)
      .def("total_submits", &lobx::ConcurrentOrderBook::total_submits)
      .def("total_cancels", &lobx::ConcurrentOrderBook::total_cancels)
      .def("total_fills", &lobx::ConcurrentOrderBook::total_fills)
      .def("total_traded_volume", &lobx::ConcurrentOrderBook::total_traded_volume);

  py::class_<lobx::ExchangeGateway>(m, "NativeGateway")
      .def(py::init<std::size_t, std::size_t>(), py::arg("ingress_capacity") = 0,
           py::arg("egress_capacity") = 0)
      .def("start", &lobx::ExchangeGateway::start)
      .def("stop", &lobx::ExchangeGateway::stop)
      .def("is_running", &lobx::ExchangeGateway::is_running)
      .def(
          "submit_order",
          [](lobx::ExchangeGateway& gw, std::int64_t id, std::int64_t agent, int side, int type,
             std::optional<std::int64_t> price, std::int64_t qty, std::int64_t ts) {
            return gw.submit_order(lobx::Order{
                id, agent, side == 0 ? lobx::Side::Buy : lobx::Side::Sell,
                type == 0 ? lobx::OrderType::Limit : lobx::OrderType::Market, price, qty, ts});
          },
          py::arg("id"), py::arg("agent"), py::arg("side"), py::arg("type"), py::arg("price"),
          py::arg("qty"), py::arg("ts"))
      .def("cancel_order", &lobx::ExchangeGateway::cancel_order, py::arg("order_id"),
           py::arg("ts"), py::arg("agent_id"))
      .def(
          "wait_until_processed",
          [](lobx::ExchangeGateway& gw, uint64_t seq, int timeout_ms) {
            return gw.wait_until_processed(seq, std::chrono::milliseconds(timeout_ms));
          },
          py::arg("seq_id"), py::arg("timeout_ms") = 5000,
          py::call_guard<py::gil_scoped_release>())
      .def("poll_event",
           [](lobx::ExchangeGateway& gw) -> std::optional<EventTuple> {
             auto ev = gw.poll_event();
             if (!ev) return std::nullopt;
             return std::make_tuple(ev->seq_id, static_cast<int>(ev->type), ev->order_id,
                                    ev->agent_id, ev->price, ev->qty, ev->ts, ev->success);
           })
      .def("total_requests_received", &lobx::ExchangeGateway::total_requests_received)
      .def("total_requests_processed", &lobx::ExchangeGateway::total_requests_processed)
      .def("total_trades_published", &lobx::ExchangeGateway::total_trades_published);
}
