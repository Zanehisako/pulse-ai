/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_BASE_URL?: string;
  readonly VITE_API_URL?: string;
  readonly VITE_WS_BASE_URL?: string;
  readonly VITE_DEFAULT_MODEL?: string;
  readonly VITE_API_STATUS_PATH?: string;
  readonly VITE_API_STREAM_PATH?: string;
  readonly VITE_WS_PATH?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
