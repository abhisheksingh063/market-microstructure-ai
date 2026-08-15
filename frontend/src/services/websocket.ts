type MessageHandler = (data: unknown) => void;

export class WebSocketClient {
  private ws: WebSocket | null = null;
  private handlers = new Map<string, Set<MessageHandler>>();
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private connectionListeners = new Set<(connected: boolean) => void>();
  private url: string;
  private connected = false;

  constructor(url: string = `ws://${location.host}/ws`) {
    this.url = url;
  }

  get isConnected() {
    return this.connected;
  }

  onConnectionChange(listener: (connected: boolean) => void) {
    this.connectionListeners.add(listener);
    listener(this.connected);
  }

  offConnectionChange(listener: (connected: boolean) => void) {
    this.connectionListeners.delete(listener);
  }

  connect() {
    if (this.ws?.readyState === WebSocket.OPEN) return;
    this.ws = new WebSocket(this.url);

    this.ws.onopen = () => {
      this.connected = true;
      this.notifyConnectionChange();
    };

    this.ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        const handlers = this.handlers.get(msg.type);
        if (handlers) {
          handlers.forEach((fn) => fn(msg.payload));
        }
      } catch {
        console.warn("Invalid WebSocket message:", event.data);
      }
    };

    this.ws.onclose = () => {
      this.connected = false;
      this.notifyConnectionChange();
      this.scheduleReconnect();
    };

    this.ws.onerror = () => {
      this.ws?.close();
    };
  }

  on(type: string, handler: MessageHandler) {
    if (!this.handlers.has(type)) {
      this.handlers.set(type, new Set());
    }
    this.handlers.get(type)!.add(handler);
  }

  off(type: string, handler: MessageHandler) {
    this.handlers.get(type)?.delete(handler);
  }

  disconnect() {
    if (this.reconnectTimer) clearTimeout(this.reconnectTimer);
    this.ws?.close();
    this.ws = null;
  }

  private scheduleReconnect() {
    this.reconnectTimer = setTimeout(() => this.connect(), 3000);
  }

  private notifyConnectionChange() {
    this.connectionListeners.forEach((listener) => listener(this.connected));
  }
}

export const wsClient = new WebSocketClient();
