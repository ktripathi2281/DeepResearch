import { ResearchWorkspace } from "../components/ResearchWorkspace";

export default function Home() {
  return (
    <div className="page">
      <header className="site-header">
        <h1 className="site-title">DeepResearch</h1>
        <p className="site-tagline">Evidence-based research assistant</p>
        <p className="site-lede">
          Ask a complex question → DeepResearch investigates → evidence-backed answer with
          citations.
        </p>
      </header>
      <main className="site-main">
        <ResearchWorkspace />
      </main>
      <footer className="site-footer">
        <p>
          Local-first research over an indexed corpus. Answers cite only retrieved evidence;
          request IDs make every investigation traceable.
        </p>
      </footer>
    </div>
  );
}
