import { useRef } from "react";
import Image from "next/image";
import Link from "next/link";
import useScrollReveal from "../../hooks/useScrollReveal";

interface Product {
  id: string;
  name: string;
  tagline: string;
  description: string;
  imageSrc: string;
  blurDataURL: string;
  features: string[];
}

const PRODUCTS: Product[] = [
  {
    id: "glucowave",
    name: "Smart OCR Ingestion Core",
    tagline: "Prescription Digitisation Module",
    description: "An advanced machine-learning scanning pipeline that converts handwritten, printed, or digital prescription formats into structured medicine schemas using hybrid NLP parsing.",
    imageSrc: "/products/glucowave.webp",
    // Tiny inline 8x8 base64 WebP image for beautiful blurred loaders
    blurDataURL: "data:image/webp;base64,UklGRmAAAABXRUJQVlA4IFYAAAAwAQCdASoIAAgAAkA4JaACdAE4AP7/4v/bAAD++uEAAAA=",
    features: [
      "Handwritten prescription note digitization",
      "Intelligent dosage and frequency extraction",
      "Automated clinical abbreviation calibration"
    ]
  },
  {
    id: "touchwave",
    name: "Interactive DDI Node Network",
    tagline: "Real-time Molecular Hazard Diagnostics",
    description: "A high-performance molecular vector network that automatically maps prescription substances, checking for severe bleeding risks, toxicity, and moderate warning dynamics.",
    imageSrc: "/products/touchwave.webp",
    blurDataURL: "data:image/webp;base64,UklGRmAAAABXRUJQVlA4IFYAAAAwAQCdASoIAAgAAkA4JaACdAE4AP7/4v/bAAD++uEAAAA=",
    features: [
      "Molecular node interaction maps",
      "Real-time hazard scoring and alerts",
      "Integrated hospital compliance guidelines"
    ]
  },
  {
    id: "mpvt",
    name: "Symmetrical Clinical Predictor Hub",
    tagline: "Condition Inference Engine",
    description: "A powerful semantic vector retriever mapping drug groupings to probable patient indications, providing clinician boards with explainable primary condition predictions.",
    imageSrc: "/products/mpvt.webp",
    blurDataURL: "data:image/webp;base64,UklGRmAAAABXRUJQVlA4IFYAAAAwAQCdASoIAAgAAkA4JaACdAE4AP7/4v/bAAD++uEAAAA=",
    features: [
      "Condition likelihood predictions",
      "Semantic vector database indexers",
      "Exportable safety and compliance reports"
    ]
  }
];

export default function ProductShowcase() {
  const sectionRef = useRef<HTMLElement>(null);
  useScrollReveal(sectionRef);

  return (
    <section 
      ref={sectionRef}
      id="modules" 
      style={{
        padding: "var(--section-spacing) 24px",
        backgroundColor: "var(--bg-secondary)",
        position: "relative",
        overflow: "hidden"
      }}
      aria-labelledby="products-title"
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
            marginBottom: "80px" 
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
            Core Technologies
          </span>
          <h2 
            id="products-title"
            style={{
              fontSize: "var(--font-h2)",
              fontWeight: 700,
              color: "var(--text-primary)",
              letterSpacing: "var(--letter-spacing-tight)"
            }}
          >
            Primary MediIntel Technology Modules
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


        {/* Product Cards Row */}
        <div 
          style={{
            display: "grid",
            gridTemplateColumns: "1fr",
            gap: "32px"
          }}
          className="showcase-grid"
        >
          {PRODUCTS.map((product, index) => (
            <div 
              key={product.id}
              className="scroll-reveal hover-premium"
              style={{
                background: "var(--glass-bg)",
                border: "var(--glass-border)",
                borderRadius: "var(--glass-radius-lg)",
                backdropFilter: "var(--glass-blur)",
                boxShadow: "var(--glass-shadow)",
                display: "grid",
                gridTemplateColumns: "1fr",
                overflow: "hidden",
                alignItems: "center",
                transition: "all 0.3s ease"
              }}
            >
              {/* Product Visual */}
              <div 
                style={{ 
                  position: "relative", 
                  width: "100%", 
                  height: "360px",
                  background: "rgba(0,0,0,0.2)"
                }}
              >
                <Image
                  src={product.imageSrc}
                  alt={`${product.name} - ${product.tagline}`}
                  placeholder="blur"
                  blurDataURL={product.blurDataURL}
                  fill
                  sizes="(max-width: 768px) 100vw, (max-width: 1280px) 50vw, 33vw"
                  loading="lazy"
                  style={{
                    objectFit: "cover",
                    transition: "transform 0.5s ease"
                  }}
                  className="product-image"
                />
              </div>

              {/* Product Contents */}
              <div 
                style={{ 
                  padding: "48px 32px",
                  display: "flex",
                  flexDirection: "column",
                  justifyContent: "center"
                }}
              >
                <div 
                  style={{
                    fontSize: "14px",
                    fontWeight: 700,
                    color: "var(--accent-primary)",
                    marginBottom: "8px",
                    textTransform: "uppercase",
                    letterSpacing: "0.1em"
                  }}
                >
                  {product.tagline}
                </div>
                <h3 
                  style={{
                    fontSize: "var(--font-h3)",
                    fontWeight: 700,
                    color: "var(--text-primary)",
                    marginBottom: "16px"
                  }}
                >
                  {product.name}
                </h3>
                <p 
                  style={{
                    color: "var(--neutral)",
                    fontSize: "16px",
                    lineHeight: 1.6,
                    marginBottom: "28px"
                  }}
                >
                  {product.description}
                </p>

                {/* Features List */}
                <ul 
                  style={{ 
                    listStyle: "none", 
                    display: "grid", 
                    gap: "12px", 
                    marginBottom: "36px" 
                  }}
                  aria-label={`Key features of ${product.name}`}
                >
                  {product.features.map((feature, i) => (
                    <li 
                      key={i} 
                      style={{ 
                        display: "flex", 
                        alignItems: "center", 
                        gap: "12px",
                        color: "var(--text-primary)",
                        fontSize: "15px"
                      }}
                    >
                      <span 
                        style={{
                          width: "8px",
                          height: "8px",
                          borderRadius: "50%",
                          background: "var(--accent-soft)",
                          boxShadow: "0 0 10px var(--accent-primary)"
                        }} 
                      />
                      {feature}
                    </li>
                  ))}
                </ul>

                <Link 
                  href="/register" 
                  className="btn-hover"
                  style={{
                    alignSelf: "flex-start",
                    border: "1px solid rgba(255,255,255,0.2)",
                    padding: "12px 28px",
                    borderRadius: "999px",
                    color: "var(--text-primary)",
                    fontWeight: 600,
                    fontSize: "14px",
                    backdropFilter: "var(--glass-blur)",
                    background: "rgba(255,255,255,0.02)"
                  }}
                >
                  Request Technical Info
                </Link>
              </div>
            </div>
          ))}
        </div>
      </div>

      <style jsx global>{`
        @media (min-width: 1024px) {
          .showcase-grid {
            gap: 48px !important;
          }
          .hover-premium {
            grid-template-columns: 1fr 1fr !important;
          }
          .hover-premium:nth-child(even) > div:first-child {
            order: 2 !important;
          }
        }
        .hover-premium:hover .product-image {
          transform: scale(1.05);
        }
      `}</style>
    </section>
  );
}
