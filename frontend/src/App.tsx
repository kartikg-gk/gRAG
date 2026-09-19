import { ClerkProvider, SignIn, SignUp, UserButton, useAuth, useUser } from "@clerk/clerk-react";
import { useLayoutEffect } from "react";
import { BrowserRouter, Link, Route, Routes } from "react-router-dom";
import { Toaster } from "sonner";

import { AppShell } from "@/components/shell/AppShell";
import { AuthUiProvider, type AuthUiState } from "@/contexts/AuthUiContext";
import { StudioProvider } from "@/contexts/StudioContext";
import { setTokenGetter } from "@/lib/authToken";
import { HomePage } from "@/pages/HomePage";
import { GraphsPage, QueriesPage, SessionsPage, SettingsPage, TracesPage } from "@/pages/StudioPages";

const clerkPublishableKey: string | undefined = import.meta.env.VITE_CLERK_PUBLISHABLE_KEY;

function StudioRoutes() {
  return (
    <Routes>
      <Route element={<AppShell />}>
        <Route index element={<HomePage />} />
        <Route path="studio" element={<HomePage />} />
        <Route path="graph" element={<HomePage />} />
        <Route path="traces" element={<TracesPage />} />
        <Route path="queries" element={<QueriesPage />} />
        <Route path="graphs" element={<GraphsPage />} />
        <Route path="sessions" element={<SessionsPage />} />
        <Route path="settings" element={<SettingsPage />} />
      </Route>
    </Routes>
  );
}

function AuthPage({ mode }: { mode: "sign-in" | "sign-up" }) {
  return (
    <main className="flex min-h-dvh items-center justify-center overflow-y-auto bg-paper p-4 text-ink">
      {mode === "sign-in" ? (
        <SignIn routing="path" path="/sign-in" signUpUrl="/sign-up" fallbackRedirectUrl="/" />
      ) : (
        <SignUp routing="path" path="/sign-up" signInUrl="/sign-in" fallbackRedirectUrl="/" />
      )}
    </main>
  );
}

function ClerkApplication() {
  const { getToken } = useAuth();
  const { isLoaded, user } = useUser();

  useLayoutEffect(() => {
    setTokenGetter(() => getToken());
    return () => setTokenGetter(null);
  }, [getToken]);

  const authUi: AuthUiState = {
    enabled: true,
    ready: isLoaded,
    control: !isLoaded ? (
      <span className="whitespace-nowrap text-xs text-ink-muted" role="status" aria-live="polite">Account…</span>
    ) : user ? (
      <UserButton afterSignOutUrl="/" />
    ) : (
      <Link className="page-action whitespace-nowrap" to="/sign-in">Sign in</Link>
    ),
  };

  return (
    <BrowserRouter>
      <Routes>
        <Route path="/sign-in/*" element={<AuthPage mode="sign-in" />} />
        <Route path="/sign-up/*" element={<AuthPage mode="sign-up" />} />
        <Route
          path="*"
          element={(
            <AuthUiProvider value={authUi}>
              <StudioProvider identity={{ userId: user?.id ?? null, email: user?.primaryEmailAddress?.emailAddress ?? null, ready: isLoaded }}>
                <StudioRoutes />
              </StudioProvider>
            </AuthUiProvider>
          )}
        />
      </Routes>
      <Toaster theme="system" position="bottom-right" />
    </BrowserRouter>
  );
}

function DisabledApplication() {
  return (
    <BrowserRouter>
      <AuthUiProvider value={{ enabled: false, ready: true, control: null }}>
        <StudioProvider identity={{ userId: null, email: null, ready: true }}>
          <StudioRoutes />
        </StudioProvider>
      </AuthUiProvider>
      <Toaster theme="system" position="bottom-right" />
    </BrowserRouter>
  );
}

export default function App() {
  if (!clerkPublishableKey) return <DisabledApplication />;
  return <ClerkProvider publishableKey={clerkPublishableKey} afterSignOutUrl="/"><ClerkApplication /></ClerkProvider>;
}
