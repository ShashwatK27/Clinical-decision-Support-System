import { useRef } from "react";
import useScrollReveal from "../../hooks/useScrollReveal";

interface Milestone {
  year: string;
  title: string;
  achievement: string;
  description: string;
}

const MILESTONES: Milestone[] = [
  {
    year: "2023",
    title: "CDSS Platform Blueprint",
    achievement: "Core Logic Initialization",
    description: "Architected the hybrid NLP prescription parsing schemas and mapped initial vector indexes to disease symptom databases."
  },
  {
    year: "2024",
    title: "Intelligent OCR Release",
    achievement: "Handwritten Recognition Benchmarks",
    description: "Introduced image pre-processing thresholds, achieving a verified handwritten clinical OCR accuracy rating of 98.4%."
  },
  {
    year: "2025",
    title: "Molecular Vector DDI",
    achievement: "Hazard Mapping Integration",
    description: "Connected relational SQL schemas and molecular check metrics to automatically trace severe bleeding hazards and drug-to-condition vectors."
  },
  {
    year: "2026",
    title: "Live Symmetrical Cockpit",
    achievement: "Next-Gen Platform Launch",
    description: "Released the interactive clinical console dashboard featuring live WebGL diagnostics, rate-limited form handlers, and DB integrations."
  }
];

export default function Awards() {
  const sectionRef = useRef<HTMLElement>(null);
  useScrollReveal(sectionRef);

  return (
    <section 
      ref={sectionRef}
      id="awards"
      style={{
        padding: "var(--section-spacing) 24px",
        backgroundColor: "var(--bg-secondary)",
        position: "relative",
        overflow: "hidden"
      }}
      aria-labelledby="awards-title"
    >
      <div 
        style={{
          width: "100%",
          maxWidth: "var(--content-max-width)",
          margin: "0 auto"
        }}
      >
        {/* Section Header */}
        <div 
          className="scroll-reveal"
          style={{ 
            textAlign: "center", 
            marginBottom: "64px" 
          }}
        >
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
            Development Roadmap
          </span>
          <h2 
            id="awards-title"
            style={{
              fontSize: "var(--font-h2)",
              fontWeight: 700,
              color: "var(--text-primary)",
              letterSpacing: "var(--letter-spacing-tight)"
            }}
          >
            System Evolution &amp; Benchmarks
          </h2>
          <div 
            style={{
              width: "80px",
              height: "2px",
              background: "linear-gradient(90deg, var(--accent-primary) 0%, transparent 100%)",
              margin: "24px auto 0"
            }}
          />
        </div>


        {/* Horizontal Timeline Scroll Container */}
        <div 
          className="scroll-reveal timeline-scroll"
          style={{
            display: "flex",
            gap: "28px",
            overflowX: "auto",
            paddingBottom: "32px",
            scrollSnapType: "x mandatory",
            WebkitOverflowScrolling: "touch",
            scrollbarWidth: "thin"
          }}
          role="region"
          aria-label="Awards and Milestones Horizontal Scroll"
        >
          {MILESTONES.map((item, index) => (
            <div 
              key={index}
              style={{
                flex: "0 0 300px",
                scrollSnapAlign: "start",
                background: "var(--glass-bg)",
                border: "var(--glass-border)",
                borderRadius: "var(--glass-radius)",
                padding: "36px 28px",
                backdropFilter: "var(--glass-blur)",
                boxShadow: "var(--glass-shadow)",
                position: "relative",
                display: "flex",
                flexDirection: "column"
              }}
              tabIndex={0}
            >
              {/* Year Badge */}
              <div 
                style={{
                  fontSize: "36px",
                  fontWeight: 800,
                  color: "var(--accent-primary)",
                  marginBottom: "18px",
                  lineHeight: 1
                }}
              >
                {item.year}
              </div>

              {/* Title & Achievement */}
              <h3 
                style={{
                  fontSize: "18px",
                  fontWeight: 700,
                  color: "var(--text-primary)",
                  marginBottom: "6px"
                }}
              >
                {item.title}
              </h3>
              <span 
                style={{
                  fontSize: "12px",
                  fontWeight: 600,
                  color: "var(--accent-soft)",
                  display: "block",
                  marginBottom: "16px",
                  textTransform: "uppercase",
                  letterSpacing: "0.05em"
                }}
              >
                {item.achievement}
              </span>

              {/* Description */}
              <p 
                style={{
                  fontSize: "14px",
                  color: "var(--neutral)",
                  lineHeight: 1.6,
                  margin: 0
                }}
              >
                {item.description}
              </p>

              {/* Connecting Dot/Line visual */}
              <div 
                style={{
                  position: "absolute",
                  top: "36px",
                  right: "-14px",
                  width: "28px",
                  height: "2px",
                  background: "linear-gradient(90deg, var(--accent-primary) 0%, transparent 100%)",
                  display: index === MILESTONES.length - 1 ? "none" : "block"
                }} 
              />
            </div>
          ))}
        </div>
      </div>

      <style jsx global>{`
        /* Style adjustments for custom timeline scrollbars */
        .timeline-scroll::-webkit-scrollbar {
          height: 6px !important;
        }
        .timeline-scroll::-webkit-scrollbar-thumb {
          background: rgba(255, 255, 255, 0.08) !important;
          border-radius: 99px !important;
        }
        .timeline-scroll::-webkit-scrollbar-thumb:hover {
          background: var(--accent-primary) !important;
        }
      `}</style>
    </section>
  );
}
