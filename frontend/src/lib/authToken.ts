type TokenGetter = () => Promise<string | null> | string | null;

let getter: TokenGetter | null = null;

export function setTokenGetter(fn: TokenGetter | null): void {
  getter = fn;
}

export async function getAuthToken(): Promise<string | null> {
  if (!getter) return null;
  try {
    return (await getter()) ?? null;
  } catch {
    return null;
  }
}
