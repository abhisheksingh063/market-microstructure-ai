import { useEffect } from "react";
import { wsClient } from "../services/websocket";

export function useWebSocket(
  type: string,
  handler: (data: unknown) => void
) {
  useEffect(() => {
    wsClient.on(type, handler);
    return () => wsClient.off(type, handler);
  }, [type, handler]);
}
