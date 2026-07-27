import { BrowserRouter, Routes, Route, Link } from "react-router-dom";
import { Dashboard } from "./pages/Dashboard";
import { OrderBookPage } from "./pages/OrderBookPage";
import { Charts } from "./pages/Charts";
import { SimulationPage } from "./pages/SimulationPage";
import { Agents } from "./pages/Agents";
import { Trades } from "./pages/Trades";
import { RLTraining } from "./pages/RLTraining";
import { Evaluation } from "./pages/Evaluation";
import "./App.css";

function Navbar() {
  const links = [
    { to: "/", label: "Dashboard" },
    { to: "/orderbook", label: "Order Book" },
    { to: "/charts", label: "Charts" },
    { to: "/simulation", label: "Simulation" },
    { to: "/agents", label: "Agents" },
    { to: "/trades", label: "Trades" },
    { to: "/rl", label: "RL Training" },
    { to: "/evaluation", label: "Evaluation" },
  ];

  return (
    <nav>
      {links.map((l) => (
        <Link key={l.to} to={l.to}>
          {l.label}
        </Link>
      ))}
    </nav>
  );
}

function App() {
  return (
    <BrowserRouter>
      <Navbar />
      <main>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/orderbook" element={<OrderBookPage />} />
          <Route path="/charts" element={<Charts />} />
          <Route path="/simulation" element={<SimulationPage />} />
          <Route path="/agents" element={<Agents />} />
          <Route path="/trades" element={<Trades />} />
          <Route path="/rl" element={<RLTraining />} />
          <Route path="/evaluation" element={<Evaluation />} />
        </Routes>
      </main>
    </BrowserRouter>
  );
}

export default App;
