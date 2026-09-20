import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "DeepResearch — Evidence-based research assistant",
  description:
    "Ask a complex question. DeepResearch investigates the indexed corpus and answers with evidence-backed citations.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
