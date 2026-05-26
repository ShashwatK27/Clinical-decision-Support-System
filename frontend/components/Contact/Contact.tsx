import { useState, FormEvent, ChangeEvent, useRef } from "react";
import useScrollReveal from "../../hooks/useScrollReveal";

interface FormState {
  name: string;
  email: string;
  subject: string;
  message: string;
}

export default function Contact() {
  const sectionRef = useRef<HTMLElement>(null);
  useScrollReveal(sectionRef);

  const [formData, setFormData] = useState<FormState>({
    name: "",
    email: "",
    subject: "",
    message: ""
  });

  const [status, setStatus] = useState<"idle" | "loading" | "success" | "error" | "rate_limited">("idle");
  const [errorMessage, setErrorMessage] = useState<string>("");

  const handleChange = (e: ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    
    // Core Client-Side Validation
    if (!formData.name || !formData.email || !formData.message) {
      setStatus("error");
      setErrorMessage("Please complete all required fields (*).");
      return;
    }

    setStatus("loading");
    setErrorMessage("");

    try {
      const response = await fetch("/api/contact", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify(formData)
      });

      const result = await response.json();

      if (response.status === 200) {
        setStatus("success");
        setFormData({ name: "", email: "", subject: "", message: "" });
      } else if (response.status === 429) {
        setStatus("rate_limited");
        setErrorMessage(result.error || "Rate limit reached. Please wait a minute before retrying.");
      } else {
        setStatus("error");
        setErrorMessage(result.error || "An error occurred. Please verify your entries.");
      }
    } catch (err) {
      setStatus("error");
      setErrorMessage("System timeout. Please check your connectivity and try again.");
    }
  };

  return (
    <section 
      ref={sectionRef}
      id="contact" 
      style={{
        padding: "var(--section-spacing) 24px",
        backgroundColor: "var(--bg-primary)",
        position: "relative"
      }}
      aria-labelledby="contact-title"
    >
      <div 
        style={{
          width: "100%",
          maxWidth: "var(--content-max-width)",
          margin: "0 auto",
          display: "grid",
          gridTemplateColumns: "1fr",
          gap: "56px"
        }}
        className="contact-grid"
      >
        {/* Left Column: Corporate Information */}
        <div className="scroll-reveal" style={{ display: "flex", flexDirection: "column", justifyContent: "center" }}>
          <span 
            style={{
              fontSize: "14px",
              fontWeight: 700,
              textTransform: "uppercase",
              color: "var(--accent-primary)",
              letterSpacing: "0.15em",
              display: "block",
              marginBottom: "16px"
            }}
          >
            Connect With Us
          </span>
          <h2 
            id="contact-title"
            style={{
              fontSize: "var(--font-h2)",
              fontWeight: 700,
              color: "var(--text-primary)",
              marginBottom: "24px",
              letterSpacing: "var(--letter-spacing-tight)"
            }}
          >
            Request CDSS Integration
          </h2>
          <p 
            style={{
              fontSize: "16px",
              lineHeight: 1.6,
              color: "var(--neutral)",
              marginBottom: "40px",
              maxWidth: "520px"
            }}
          >
            Interested in partner trials, hospital-wide API telemetry deployments, or custom medical knowledge base integrations? Complete our secure integration inquiry form.
          </p>

          {/* Contact Methods list */}
          <div style={{ display: "grid", gap: "28px" }} aria-label="Corporate Headquarters Info">
            <div style={{ display: "flex", gap: "20px", alignItems: "center" }}>
              <span style={{ fontSize: "28px" }} aria-hidden="true">🏢</span>
              <div>
                <strong style={{ display: "block", fontSize: "16px", color: "var(--text-primary)" }}>Platform Lab</strong>
                <span style={{ color: "var(--neutral)", fontSize: "15px" }}>CDSS AI Symmetrical Platform Core</span>
              </div>
            </div>
            <div style={{ display: "flex", gap: "20px", alignItems: "center" }}>
              <span style={{ fontSize: "28px" }} aria-hidden="true">✉️</span>
              <div>
                <strong style={{ display: "block", fontSize: "16px", color: "var(--text-primary)" }}>Support &amp; Licenses</strong>
                <span style={{ color: "var(--neutral)", fontSize: "15px" }}>support@cdss-ai.com</span>
              </div>
            </div>

          </div>
        </div>

        {/* Right Column: Secure Glass Form */}
        <div 
          className="scroll-reveal"
          style={{
            background: "var(--glass-bg)",
            border: "var(--glass-border)",
            borderRadius: "var(--glass-radius-lg)",
            padding: "48px 36px",
            backdropFilter: "var(--glass-blur)",
            boxShadow: "var(--glass-shadow)"
          }}
        >
          <form onSubmit={handleSubmit} style={{ display: "grid", gap: "24px" }} aria-describedby="form-status-msg">
            {/* Input Name */}
            <div>
              <label 
                htmlFor="contact-name" 
                style={{ display: "block", fontSize: "14px", fontWeight: 600, color: "var(--text-primary)", marginBottom: "8px" }}
              >
                Full Name *
              </label>
              <input
                id="contact-name"
                name="name"
                type="text"
                required
                value={formData.name}
                onChange={handleChange}
                disabled={status === "loading"}
                style={{
                  width: "100%",
                  background: "rgba(0,0,0,0.2)",
                  border: "1px solid rgba(255,255,255,0.08)",
                  borderRadius: "12px",
                  padding: "14px 18px",
                  color: "var(--text-primary)",
                  outline: "none",
                  transition: "border-color 0.2s ease"
                }}
              />
            </div>

            {/* Input Email */}
            <div>
              <label 
                htmlFor="contact-email" 
                style={{ display: "block", fontSize: "14px", fontWeight: 600, color: "var(--text-primary)", marginBottom: "8px" }}
              >
                Email Address *
              </label>
              <input
                id="contact-email"
                name="email"
                type="email"
                required
                value={formData.email}
                onChange={handleChange}
                disabled={status === "loading"}
                style={{
                  width: "100%",
                  background: "rgba(0,0,0,0.2)",
                  border: "1px solid rgba(255,255,255,0.08)",
                  borderRadius: "12px",
                  padding: "14px 18px",
                  color: "var(--text-primary)",
                  outline: "none",
                  transition: "border-color 0.2s ease"
                }}
              />
            </div>

            {/* Input Subject */}
            <div>
              <label 
                htmlFor="contact-subject" 
                style={{ display: "block", fontSize: "14px", fontWeight: 600, color: "var(--text-primary)", marginBottom: "8px" }}
              >
                Subject
              </label>
              <input
                id="contact-subject"
                name="subject"
                type="text"
                value={formData.subject}
                onChange={handleChange}
                disabled={status === "loading"}
                style={{
                  width: "100%",
                  background: "rgba(0,0,0,0.2)",
                  border: "1px solid rgba(255,255,255,0.08)",
                  borderRadius: "12px",
                  padding: "14px 18px",
                  color: "var(--text-primary)",
                  outline: "none",
                  transition: "border-color 0.2s ease"
                }}
              />
            </div>

            {/* Input Message */}
            <div>
              <label 
                htmlFor="contact-message" 
                style={{ display: "block", fontSize: "14px", fontWeight: 600, color: "var(--text-primary)", marginBottom: "8px" }}
              >
                Message Details *
              </label>
              <textarea
                id="contact-message"
                name="message"
                required
                rows={5}
                value={formData.message}
                onChange={handleChange}
                disabled={status === "loading"}
                style={{
                  width: "100%",
                  background: "rgba(0,0,0,0.2)",
                  border: "1px solid rgba(255,255,255,0.08)",
                  borderRadius: "12px",
                  padding: "14px 18px",
                  color: "var(--text-primary)",
                  outline: "none",
                  resize: "vertical",
                  transition: "border-color 0.2s ease"
                }}
              />
            </div>

            {/* Form Action Feedback Box */}
            <div id="form-status-msg" aria-live="polite" style={{ fontSize: "14px", fontWeight: 600 }}>
              {status === "success" && (
                <div style={{ color: "var(--accent-primary)", padding: "8px 0" }}>
                  ✓ Secure submission successful! Our medtech board will contact you shortly.
                </div>
              )}
              {status === "rate_limited" && (
                <div style={{ color: "#fbbf24", padding: "8px 0" }}>
                  ⚠️ Too many submissions. Please wait a brief moment before sending again.
                </div>
              )}
              {status === "error" && (
                <div style={{ color: "#f87171", padding: "8px 0" }}>
                  ✕ {errorMessage}
                </div>
              )}
            </div>

            {/* Submit Button */}
            <button
              type="submit"
              disabled={status === "loading"}
              className="btn-hover"
              style={{
                width: "100%",
                background: status === "loading" ? "var(--neutral)" : "var(--white)",
                color: "var(--black)",
                border: "none",
                borderRadius: "999px",
                padding: "16px",
                fontWeight: 700,
                fontSize: "16px",
                cursor: status === "loading" ? "not-allowed" : "pointer",
                transition: "background-color 0.2s ease",
                boxShadow: "0 10px 30px rgba(255,255,255,0.05)"
              }}
              aria-label="Submit secure diagnostic inquiry form"
            >
              {status === "loading" ? "Validating & Securing..." : "Send Secure Message"}
            </button>
          </form>
        </div>
      </div>

      <style jsx global>{`
        @media (min-width: 1024px) {
          .contact-grid {
            grid-template-columns: 1fr 1.1fr !important;
          }
        }
        input:focus, textarea:focus {
          border-color: var(--accent-primary) !important;
        }
      `}</style>
    </section>
  );
}
