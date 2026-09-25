import React from "react";
import ReactDOM from "react-dom/client";

import App from "./App";
import { API_BASE } from "./lib/api";
import "./index.css";

// Open the connection to the API (DNS, TCP, TLS) while the page renders, so
// the first question does not pay for it. The address is only known at build
// time, which is why this is not a tag in index.html.
try {
  const link = document.createElement("link");
  link.rel = "preconnect";
  link.href = new URL(API_BASE).origin;
  link.crossOrigin = "anonymous";
  document.head.appendChild(link);
} catch {
  // A relative or malformed base has nothing to preconnect to.
}

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
