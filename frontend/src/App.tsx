import { BrowserRouter, Routes, Route } from "react-router-dom";
import { AppLayout } from "./components/layout/AppLayout";
import { Dashboard } from "./pages/Dashboard";
import { OrderBookPage } from "./pages/OrderBookPage";
import { Charts } from "./pages/Charts";
import { SimulationPage } from "./pages/SimulationPage";
import { Agents } from "./pages/Agents";
import { Trades } from "./pages/Trades";
import { RLTraining } from "./pages/RLTraining";
import { Evaluation } from "./pages/Evaluation";
import { Settings } from "./pages/Settings";
import "./App.css";

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<AppLayout />}>
          <Route path="/" element={<Dashboard />} />
          <Route path="/orderbook" element={<OrderBookPage />} />
          <Route path="/charts" element={<Charts />} />
          <Route path="/simulation" element={<SimulationPage />} />
          <Route path="/agents" element={<Agents />} />
          <Route path="/trades" element={<Trades />} />
          <Route path="/rl" element={<RLTraining />} />
          <Route path="/evaluation" element={<Evaluation />} />
          <Route path="/settings" element={<Settings />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

export default App;
