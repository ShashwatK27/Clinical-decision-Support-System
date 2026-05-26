import { useState, useEffect } from "react";
import dynamic from "next/dynamic";
import NavBar from "../components/NavBar";
import axios from "axios";

// Dynamically load R3F components to avoid SSR canvas layout issues
const ScannerScene = dynamic(() => import("../components/ScannerScene"), { ssr: false });
const NetworkScene = dynamic(() => import("../components/NetworkScene"), { ssr: false });

// Deep Navy cockpit wrapper styling
const wrapperStyle = {
  minHeight: "100vh",
  background: "#060913",
  color: "#f8fafc",
  fontFamily: "'Outfit', 'Inter', sans-serif",
  paddingBottom: 40,
};

// Preset Patient Scenarios
const PRESETS = {
  safe: {
    name: "Scenario A: Symmetrical Safe Therapy",
    text: "Amoxicillin 500mg once daily + Acetaminophen 325mg for headache + Ibuprofen 200mg + Loratadine 10mg antihistamine",
    drugs: ["Amoxicillin", "Acetaminophen", "Ibuprofen", "Loratadine"],
    interactions: [],
    conditions: [
      { label: "Bacterial Infection", confidence: "97%" },
      { label: "Pain", confidence: "92%" },
      { label: "Allergic Rhinitis", confidence: "88%" }
    ],
    metrics: {
      detected: 4,
      checked: 6,
      severe: 0,
      confidence: "98.4%",
      time: "185 ms"
    }
  },
  severe: {
    name: "Scenario B: Critical DDI Comorbidity",
    text: "Patient prescribed warfarin 5mg daily for clot prevention. Adding methotrexate 15mg weekly for rheumatoid arthritis, and ibuprofen 400mg twice daily for joint swelling.",
    drugs: ["Warfarin", "Methotrexate", "Ibuprofen", "Amoxicillin"],
    interactions: [
      {
        drugs: ["Warfarin", "Ibuprofen"],
        severity: "severe",
        effect: "Severe bleeding hazard. Ibuprofen (NSAID) impairs platelet aggregation and causes gastrointestinal mucosal injury, multiplying anticoagulant risk.",
        recommendation: "ABANDON Ibuprofen. Substitute with Acetaminophen (Tylenol) for mild pain. Monitor INR levels closely."
      },
      {
        drugs: ["Warfarin", "Methotrexate"],
        severity: "severe",
        effect: "Hemorrhagic toxicity risk. Methotrexate displace warfarin from plasma albumin binding sites and impairs hepatic synthesis, spiking active drug circulation.",
        recommendation: "CRITICAL ALERT. Coordinate dosage reduction under close supervision. Perform weekly coagulation diagnostics."
      }
    ],
    conditions: [
      { label: "Rheumatoid Arthritis", confidence: "95%" },
      { label: "Blood Clot Prevention", confidence: "94%" },
      { label: "Pain / Swelling", confidence: "89%" }
    ],
    metrics: {
      detected: 4,
      checked: 6,
      severe: 2,
      confidence: "96.1%",
      time: "290 ms"
    }
  }
};

export default function Dashboard() {
  // Scenario States
  const [inputText, setInputText] = useState("");
  const [activePipelineStep, setActivePipelineStep] = useState<number>(0);
  const [isScanning, setIsScanning] = useState<boolean>(false);
  const [scanProgress, setScanProgress] = useState<number>(0);
  const [currentScenario, setCurrentScenario] = useState<string>("safe"); // 'safe', 'severe', or 'custom'

  // Results Bindings
  const [drugs, setDrugs] = useState<string[]>(PRESETS.safe.drugs);
  const [interactions, setInteractions] = useState<any[]>(PRESETS.safe.interactions);
  const [conditions, setConditions] = useState<any[]>(PRESETS.safe.conditions);
  const [metrics, setMetrics] = useState<any>(PRESETS.safe.metrics);

  // Selected DDI Sidebar Detail
  const [selectedDDI, setSelectedDDI] = useState<any>(null);

  // Custom user search handler
  const handleCustomSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputText.trim()) return;

    setCurrentScenario("custom");
    setSelectedDDI(null);
    triggerScanSequence();

    // Trigger mock parser logic for custom text
    setTimeout(() => {
      // Fuzzy detect common medicines from custom text
      const rawText = inputText.toLowerCase();
      const detected: string[] = [];
      if (rawText.includes("warfarin")) detected.push("Warfarin");
      if (rawText.includes("ibuprofen") || rawText.includes("iboprofen")) detected.push("Ibuprofen");
      if (rawText.includes("methotrexate")) detected.push("Methotrexate");
      if (rawText.includes("amoxicillin") || rawText.includes("amox")) detected.push("Amoxicillin");
      if (rawText.includes("acetaminophen") || rawText.includes("paracetamol")) detected.push("Acetaminophen");
      if (rawText.includes("loratadine") || rawText.includes("claritin")) detected.push("Loratadine");

      // Default fallback if typing arbitrary things
      if (detected.length === 0) {
        detected.push("Ibuprofen", "Acetaminophen");
      }

      // Check if severe pairs overlap
      const foundIxs: any[] = [];
      if (detected.includes("Warfarin") && detected.includes("Ibuprofen")) {
        foundIxs.push(PRESETS.severe.interactions[0]);
      }
      if (detected.includes("Warfarin") && detected.includes("Methotrexate")) {
        foundIxs.push(PRESETS.severe.interactions[1]);
      }

      const predicted = [
        { label: "Symptomatic Pain", confidence: "88%" },
        { label: "Primary Indication", confidence: "80%" }
      ];

      setDrugs(detected);
      setInteractions(foundIxs);
      setConditions(predicted);
      setMetrics({
        detected: detected.length,
        checked: Math.max(1, (detected.length * (detected.length - 1)) / 2),
        severe: foundIxs.length,
        confidence: "94.2%",
        time: "240 ms"
      });
    }, 1500);
  };

  // Preset Scenario Activation Handler
  const handleLoadPreset = (type: "safe" | "severe") => {
    setCurrentScenario(type);
    setSelectedDDI(null);
    setInputText("");
    triggerScanSequence();
    
    setTimeout(() => {
      const data = PRESETS[type];
      setDrugs(data.drugs);
      setInteractions(data.interactions);
      setConditions(data.conditions);
      setMetrics(data.metrics);
    }, 1500);
  };

  // Laser Sweep & Pipeline Timeline Steps Interval
  const triggerScanSequence = () => {
    setIsScanning(true);
    setScanProgress(0);
    setActivePipelineStep(1); // Start OCR

    const duration = 1500; // 1.5 seconds total scan time
    const intervalTime = 50;
    const steps = duration / intervalTime;
    let currentStep = 0;

    const interval = setInterval(() => {
      currentStep++;
      const progress = Math.min(100, Math.round((currentStep / steps) * 100));
      setScanProgress(progress);

      // Distribute 5-stage pipeline steps evenly
      if (progress >= 20 && progress < 40) setActivePipelineStep(2); // Normalize
      else if (progress >= 40 && progress < 60) setActivePipelineStep(3); // Validate
      else if (progress >= 60 && progress < 80) setActivePipelineStep(4); // Extract
      else if (progress >= 80 && progress < 100) setActivePipelineStep(5); // Ready

      if (currentStep >= steps) {
        clearInterval(interval);
        setIsScanning(false);
        setScanProgress(100);
        // Delay pipeline active clear
        setTimeout(() => setActivePipelineStep(0), 400);
      }
    }, intervalTime);
  };

  // Automatically load selected interaction details in sidebar when list changes
  useEffect(() => {
    if (interactions.length > 0) {
      setSelectedDDI(interactions[0]);
    } else {
      setSelectedDDI(null);
    }
  }, [interactions]);

  return (
    <div className="cockpit-theme" style={wrapperStyle}>
      <NavBar />

      <main className="container" style={{ padding: "16px 24px" }}>
        {/* Preset Patient Case Scenarios Header Selector */}
        <section 
          className="cockpit-panel" 
          style={{ 
            marginBottom: 24, 
            display: "flex", 
            justifyContent: "space-between", 
            alignItems: "center",
            padding: "16px 24px",
            border: "1px solid rgba(6, 182, 212, 0.2)",
            boxShadow: "0 0 20px rgba(6, 182, 212, 0.05)"
          }}
        >
          <div>
            <h3 style={{ margin: 0, fontSize: 13, letterSpacing: 1.5, textTransform: "uppercase", color: "#38bdf8" }}>
              🧬 Clinical Presets Controls
            </h3>
            <span style={{ fontSize: 11, color: "rgba(255,255,255,0.4)" }}>
              One-click setup to compare clinical outcomes and trigger full interactive WebGL scans.
            </span>
          </div>
          <div style={{ display: "flex", gap: 14 }}>
            <button 
              className={`hud-btn ${currentScenario === "safe" ? "" : "secondary"}`}
              style={{ 
                borderWidth: currentScenario === "safe" ? 2 : 1,
                borderColor: currentScenario === "safe" ? "#10b981" : "rgba(6, 182, 212, 0.2)",
                background: currentScenario === "safe" ? "rgba(16, 185, 129, 0.15)" : ""
              }}
              onClick={() => handleLoadPreset("safe")}
            >
              🟢 Preset Safe Regimen
            </button>
            <button 
              className={`hud-btn ${currentScenario === "severe" ? "" : "secondary"}`}
              style={{ 
                borderWidth: currentScenario === "severe" ? 2 : 1,
                borderColor: currentScenario === "severe" ? "#ef4444" : "rgba(6, 182, 212, 0.2)",
                background: currentScenario === "severe" ? "rgba(239, 68, 68, 0.15)" : ""
              }}
              onClick={() => handleLoadPreset("severe")}
            >
              🔴 Preset High-Risk DDI
            </button>
          </div>
        </section>

        {/* Dynamic HUD Keyboard Intake Input Section */}
        <section className="cockpit-panel" style={{ marginBottom: 24 }}>
          <form onSubmit={handleCustomSubmit} style={{ display: "flex", gap: 16 }}>
            <input 
              className="hud-input"
              placeholder="Or type custom patient comorbidity (e.g., 'warfarin 5mg daily + ibuprofen 400mg twice daily'...) and press cyber-scan"
              type="text"
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
            />
            <button 
              className="hud-btn" 
              type="submit" 
              disabled={isScanning}
              style={{ display: "flex", alignItems: "center", gap: 8 }}
            >
              <span>⚡</span> INTIATE CYBER SCAN
            </button>
          </form>
        </section>

        {/* Symmetrical Dual-Panel Layout Grid */}
        <div style={{ display: "grid", gridTemplateColumns: "1.2fr 1fr 0.8fr", gap: 24, alignItems: "stretch" }}>
          
          {/* LEFT PANEL: Holographic Prescription Scanner Bed */}
          <div className="cockpit-panel" style={{ display: "flex", flexDirection: "column", justifyContent: "space-between" }}>
            <div>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
                <h2 style={{ margin: 0, fontSize: 16, fontWeight: 800, letterSpacing: 1 }}>
                  🛰️ HOLOGRAPHIC PRESCRIPTION SCANNER
                </h2>
                {isScanning && (
                  <span style={{ fontSize: 10, fontWeight: 700, color: "#06b6d4", background: "rgba(6, 182, 212, 0.15)", padding: "3px 8px", borderRadius: 99 }}>
                    SCANNING {scanProgress}%
                  </span>
                )}
              </div>
              <p style={{ margin: 0, fontSize: 12, color: "rgba(255,255,255,0.5)", marginBottom: 18 }}>
                Horizontal laser plane and particles track OCR anomalies inside the WebGL workspace.
              </p>

              {/* 5-Step Pipeline Progress HUD */}
              <div className="pipeline-container">
                <div className={`pipeline-step ${activePipelineStep === 1 ? "active" : activePipelineStep > 1 ? "completed" : ""}`}>
                  <span>1. OCR</span>
                </div>
                <div className={`pipeline-step ${activePipelineStep === 2 ? "active" : activePipelineStep > 2 ? "completed" : ""}`}>
                  <span>2. Normalise</span>
                </div>
                <div className={`pipeline-step ${activePipelineStep === 3 ? "active" : activePipelineStep > 3 ? "completed" : ""}`}>
                  <span>3. Validate</span>
                </div>
                <div className={`pipeline-step ${activePipelineStep === 4 ? "active" : activePipelineStep > 4 ? "completed" : ""}`}>
                  <span>4. Extract</span>
                </div>
                <div className={`pipeline-step ${activePipelineStep === 5 ? "active" : activePipelineStep > 5 ? "completed" : ""}`}>
                  <span>5. Ready</span>
                </div>
              </div>
            </div>

            {/* 3D WebGL Canvas Bed */}
            <div style={{ 
              height: 380, 
              background: "radial-gradient(circle at center, rgba(10, 24, 50, 0.4) 0%, rgba(5, 8, 16, 0.8) 100%)", 
              borderRadius: 14, 
              border: "1px solid rgba(255, 255, 255, 0.05)",
              overflow: "hidden" 
            }}>
              <ScannerScene isScanning={isScanning} scanProgress={scanProgress} />
            </div>

            {/* Glowing neon progress line below canvas */}
            <div style={{ marginTop: 14 }}>
              <div style={{ height: 4, width: "100%", background: "rgba(255,255,255,0.05)", borderRadius: 99, overflow: "hidden" }}>
                <div style={{ height: "100%", width: `${scanProgress}%`, background: "#06b6d4", boxShadow: "0 0 10px #06b6d4", transition: "width 0.1s ease" }} />
              </div>
            </div>
          </div>

          {/* RIGHT PANEL: 3D Drug-Drug Interaction Node Network */}
          <div className="cockpit-panel" style={{ display: "flex", flexDirection: "column", justifyContent: "space-between" }}>
            <div>
              <h2 style={{ margin: 0, fontSize: 16, fontWeight: 800, letterSpacing: 1, marginBottom: 4 }}>
                🛡️ INTERACTION NODE NETWORK
              </h2>
              <p style={{ margin: 0, fontSize: 12, color: "rgba(255, 255, 255, 0.5)", marginBottom: 18 }}>
                Rotate physical glass containment jar to inspect molecular nodes and pulsing warning vectors.
              </p>
            </div>

            {/* 3D WebGL Node Canvas */}
            <div style={{ 
              height: 380, 
              background: "radial-gradient(circle at center, rgba(10, 24, 50, 0.4) 0%, rgba(5, 8, 16, 0.8) 100%)", 
              borderRadius: 14, 
              border: "1px solid rgba(255, 255, 255, 0.05)",
              overflow: "hidden" 
            }}>
              <NetworkScene 
                drugs={drugs} 
                interactions={interactions} 
                onSelectInteraction={(pair, severity, effect) => {
                  setSelectedDDI({ drugs: pair, severity, effect });
                }} 
              />
            </div>

            {/* Nodes Legend */}
            <div style={{ display: "flex", gap: 16, fontSize: 11, fontWeight: 700, padding: "8px 4px 0", letterSpacing: 0.5 }}>
              <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                <span style={{ width: 8, height: 8, borderRadius: "50%", background: "#10b981", boxShadow: "0 0 6px #10b981" }} />
                <span>SAFE NODES</span>
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                <span style={{ width: 8, height: 8, borderRadius: "50%", background: "#ef4444", boxShadow: "0 0 6px #ef4444" }} />
                <span>SEVERE HAZARDS</span>
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                <span style={{ width: 8, height: 8, borderRadius: "50%", background: "#f59e0b", boxShadow: "0 0 6px #f59e0b" }} />
                <span>MODERATE WARNS</span>
              </div>
            </div>
          </div>

          {/* RIGHT SIDEBAR: DDI Warning details */}
          <div className="cockpit-panel sidebar-panel">
            <h2 style={{ margin: 0, fontSize: 16, fontWeight: 800, letterSpacing: 1, borderBottom: "1px solid rgba(255,255,255,0.08)", paddingBottom: 12 }}>
              🩺 MEDICAL ANALYTICS
            </h2>

            {/* Live active warnings count header */}
            <div style={{
              background: interactions.length > 0 ? "rgba(239, 68, 68, 0.06)" : "rgba(16, 185, 129, 0.06)",
              border: `1px solid ${interactions.length > 0 ? "rgba(239, 68, 68, 0.2)" : "rgba(16, 185, 129, 0.2)"}`,
              padding: 12,
              borderRadius: 10,
              fontSize: 12,
              textAlign: "center"
            }}>
              {interactions.length > 0 ? (
                <span className="text-neon-red" style={{ fontWeight: 800 }}>
                  ⚠️ CRITICAL WARNING: {interactions.length} SEVERE DDIs
                </span>
              ) : (
                <span className="text-neon-green" style={{ fontWeight: 800 }}>
                  ✅ REGIMEN SECURE: NO DDIs DETECTED
                </span>
              )}
            </div>

            {/* Warning details view cards */}
            <div className="custom-scroll" style={{ flexGrow: 1, overflowY: "auto", display: "flex", flexDirection: "column", gap: 14, maxHeight: 310 }}>
              {interactions.length === 0 ? (
                <div style={{ padding: "20px 0", textAlign: "center", color: "rgba(255,255,255,0.3)", fontSize: 11 }}>
                  All nodes safe. Try loading the High-Risk DDI preset to inspect dangerous relationships.
                </div>
              ) : (
                interactions.map((ix, idx) => (
                  <div 
                    key={idx} 
                    className={`ddi-card ${ix.severity}`}
                    style={{ 
                      cursor: "pointer", 
                      borderWidth: selectedDDI?.drugs.join() === ix.drugs.join() ? "2px" : "1px",
                      borderColor: selectedDDI?.drugs.join() === ix.drugs.join() ? "#ef4444" : ""
                    }}
                    onClick={() => setSelectedDDI(ix)}
                  >
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
                      <span style={{ fontSize: 10, fontWeight: 800, color: "#ef4444", textTransform: "uppercase" }}>
                        {ix.severity.toUpperCase()} ALERT
                      </span>
                      <span>🚨</span>
                    </div>
                    <div style={{ fontWeight: 800, fontSize: 12, color: "#f8fafc", marginBottom: 6 }}>
                      {ix.drugs[0].toUpperCase()} ⟷ {ix.drugs[1].toUpperCase()}
                    </div>
                    <p style={{ margin: 0, fontSize: 11, color: "rgba(255,255,255,0.6)", lineHeight: 1.4, display: "-webkit-box", WebkitLineClamp: 2, WebkitBoxOrient: "vertical", overflow: "hidden" }}>
                      {ix.effect}
                    </p>
                  </div>
                ))
              )}
            </div>

            {/* Detailed Selected Warning HUD Pane */}
            {selectedDDI && (
              <div 
                className="warning-pulse"
                style={{
                  background: "rgba(239, 68, 68, 0.05)",
                  border: "1px solid rgba(239, 68, 68, 0.4)",
                  borderRadius: 12,
                  padding: 14,
                  fontSize: 11,
                  lineHeight: 1.4,
                  display: "flex",
                  flexDirection: "column",
                  gap: 8
                }}
              >
                <div style={{ fontWeight: 800, color: "#ef4444", fontSize: 11 }}>
                  PAIR COAGULATION DYNAMICS
                </div>
                <div style={{ color: "#f8fafc", fontWeight: 700 }}>
                  {selectedDDI.drugs[0]} ⟷ {selectedDDI.drugs[1]}
                </div>
                <div>
                  <strong style={{ color: "rgba(255,255,255,0.7)" }}>Pathology:</strong> {selectedDDI.effect}
                </div>
                <div>
                  <strong style={{ color: "#10b981" }}>Clinician Substitution:</strong> {selectedDDI.recommendation || "Maintain close observation."}
                </div>
                <button className="hud-btn danger" style={{ padding: "6px 12px", fontSize: 9, marginTop: 4 }}>
                  🛡️ VIEW CLINICAL DDI GUIDELINES
                </button>
              </div>
            )}
          </div>
        </div>

        {/* BOTTOM SECTION: SYSTEM INSIGHTS */}
        <section style={{ marginTop: 24, display: "grid", gridTemplateColumns: "1.2fr 1.8fr", gap: 24 }}>
          
          {/* Symmetrical predicted conditions */}
          <div className="cockpit-panel">
            <h3 style={{ margin: 0, fontSize: 14, fontWeight: 800, letterSpacing: 1, marginBottom: 16, color: "#38bdf8" }}>
              🧬 PRIMARY CLINICAL PREDICTIONS
            </h3>
            <div style={{ display: "grid", gap: 14 }}>
              {conditions.map((cond, idx) => (
                <div key={idx}>
                  <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, fontWeight: 700, marginBottom: 4 }}>
                    <span>{cond.label.toUpperCase()}</span>
                    <span style={{ color: "#06b6d4" }}>{cond.confidence}</span>
                  </div>
                  <div style={{ height: 6, width: "100%", background: "rgba(255, 255, 255, 0.05)", borderRadius: 99, overflow: "hidden" }}>
                    <div 
                      style={{ 
                        height: "100%", 
                        width: cond.confidence, 
                        background: "linear-gradient(90deg, #3b82f6 0%, #06b6d4 100%)",
                        boxShadow: "0 0 10px rgba(6, 182, 212, 0.3)",
                        transition: "width 1s ease" 
                      }} 
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Symmetrical medical guidelines + metrics grids */}
          <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>
            {/* Symmetrical metrics display row */}
            <div className="metrics-grid">
              <div className="metric-card">
                <div style={{ fontSize: 9, color: "rgba(255,255,255,0.4)", fontWeight: 700, letterSpacing: 0.5 }}>DRUGS DETECTED</div>
                <div className="metric-value text-neon-cyan">{metrics.detected}</div>
              </div>
              <div className="metric-card">
                <div style={{ fontSize: 9, color: "rgba(255,255,255,0.4)", fontWeight: 700, letterSpacing: 0.5 }}>INTERACTIONS</div>
                <div className="metric-value text-neon-blue">{metrics.checked}</div>
              </div>
              <div className="metric-card">
                <div style={{ fontSize: 9, color: "rgba(255,255,255,0.4)", fontWeight: 700, letterSpacing: 0.5 }}>SEVERE DDIs</div>
                <div className="metric-value text-neon-red">{metrics.severe}</div>
              </div>
              <div className="metric-card">
                <div style={{ fontSize: 9, color: "rgba(255,255,255,0.4)", fontWeight: 700, letterSpacing: 0.5 }}>OCR ACCURACY</div>
                <div className="metric-value text-neon-green">{metrics.confidence}</div>
              </div>
              <div className="metric-card">
                <div style={{ fontSize: 9, color: "rgba(255,255,255,0.4)", fontWeight: 700, letterSpacing: 0.5 }}>ANALYSIS LATENCY</div>
                <div className="metric-value text-neon-cyan" style={{ fontSize: 16, marginTop: 10 }}>{metrics.time}</div>
              </div>
            </div>

            {/* General Clinician Safety recommendations panel */}
            <div 
              className="cockpit-panel" 
              style={{ 
                flexGrow: 1, 
                display: "flex", 
                justifyContent: "space-between", 
                alignItems: "center",
                padding: "16px 24px"
              }}
            >
              <div>
                <h4 style={{ margin: 0, fontSize: 13, color: "#10b981", marginBottom: 4 }}>
                  🛡️ CLINICAL DECISION COMPLIANCE STATEMENT
                </h4>
                <p style={{ margin: 0, fontSize: 11, color: "rgba(255,255,255,0.5)", lineHeight: 1.4 }}>
                  Verify diagnostic observations manually. Cross-reference severe drug alerts with the institution hospital guidelines before dispatching therapeutic regimens.
                </p>
              </div>
              <button 
                className="hud-btn"
                style={{ 
                  background: "rgba(16, 185, 129, 0.08)", 
                  borderColor: "rgba(16, 185, 129, 0.25)",
                  color: "#10b981" 
                }}
              >
                📄 EXPORT COMPLIANCE REPORT
              </button>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}
