import { createContext, useContext, type ReactNode } from "react";

export interface AuthUiState {
  enabled: boolean;
  ready: boolean;
  control: ReactNode;
}

const AuthUiContext = createContext<AuthUiState>({ enabled: false, ready: true, control: null });

export function AuthUiProvider({ value, children }: { value: AuthUiState; children: ReactNode }) {
  return <AuthUiContext.Provider value={value}>{children}</AuthUiContext.Provider>;
}

export function useAuthUi() {
  return useContext(AuthUiContext);
}
