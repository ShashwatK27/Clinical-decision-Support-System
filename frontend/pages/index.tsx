import Head from "next/head";
import dynamic from "next/dynamic";
import Navbar from "../components/Navbar/Navbar";

// Keep Navbar SSR enabled to avoid Layout Shifts (CLS)
// Dynamically import all sections below the fold to achieve Initial JS <200KB
const Hero = dynamic(() => import("../components/Hero/Hero"), {
  ssr: false,
  loading: () => <div style={{ minHeight: "100vh", background: "var(--bg-primary)" }} />
});

const ProductShowcase = dynamic(() => import("../components/ProductShowcase/ProductShowcase"), {
  ssr: false
});

const Mission = dynamic(() => import("../components/Mission/Mission"), {
  ssr: false
});

const IndustryCards = dynamic(() => import("../components/IndustryCards/IndustryCards"), {
  ssr: false
});

const Contact = dynamic(() => import("../components/Contact/Contact"), {
  ssr: false
});

const BackgroundScene = dynamic(() => import("../components/BackgroundScene"), {
  ssr: false
});

export default function Home() {
  return (
    <>
      <BackgroundScene />
      <Head>
        {/* Core SEO Meta */}
        <title>MediIntel — Next-Gen Clinical Decision Support &amp; Analytics</title>
        <meta name="description" content="Pioneering Symmetrical Clinical Decision Support. Discover our intelligent handwritten prescription OCR intake core, molecular drug interaction mapping networks, and semantic condition predictors." />
        <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no" />
        <meta name="robots" content="index, follow" />
        
        {/* Open Graph / Facebook */}
        <meta property="og:type" content="website" />
        <meta property="og:title" content="MediIntel — Next-Gen Clinical Decision Support &amp; Analytics" />
        <meta property="og:description" content="Intelligent handwritten prescription OCR intakes, molecular interaction maps, and instant clinical condition predictions." />
        <meta property="og:image" content="/products/glucowave.webp" />

        {/* Twitter */}
        <meta name="twitter:card" content="summary_large_image" />
        <meta name="twitter:title" content="MediIntel — Next-Gen Clinical Decision Support &amp; Analytics" />
        <meta name="twitter:description" content="Continuous diagnostic support systems and interaction warning matrices." />
        
        {/* Favicon */}
        <link rel="icon" href="/favicon.ico" />

        {/* Google Fonts Preconnect */}
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&amp;family=Manrope:wght@500;600;700;800&amp;display=swap" rel="stylesheet" />

        {/* JSON-LD Technical Schema Metadata for MedicalOrganization and Organization */}
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{
            __html: JSON.stringify({
              "@context": "https://schema.org",
              "@graph": [
                {
                  "@type": "MedicalOrganization",
                  "@id": "https://www.mediintel.com/#organization",
                  "name": "MediIntel Symmetrical AI",
                  "url": "https://www.mediintel.com",
                  "logo": "https://www.mediintel.com/logo.png",
                  "description": "Next-generation Symmetrical Clinical Decision Support System and interaction warning console.",
                  "address": {
                    "@type": "PostalAddress",
                    "addressLocality": "Warsaw",
                    "addressCountry": "PL"
                  },
                  "contactPoint": {
                    "@type": "ContactPoint",
                    "email": "support@mediintel.com",
                    "contactType": "clinician support"
                  }
                },
                {
                  "@type": "MedicalWebPage",
                  "@id": "https://www.mediintel.com/#webpage",
                  "url": "https://www.mediintel.com",
                  "name": "MediIntel Diagnostics & Decision Analytics Hub",
                  "about": [
                    {
                      "@type": "MedicalDevice",
                      "name": "Smart OCR Core",
                      "description": "Intelligent handwritten clinical prescription text ingestion engine."
                    },
                    {
                      "@type": "MedicalDevice",
                      "name": "DDI Molecular Vector Network",
                      "description": "Molecular substance mapping and severe bleeding/toxicity hazard checker."
                    }
                  ]
                }
              ]
            })
          }}
        />

      </Head>

      {/* Skip to Content for Keyboard Users */}
      <a href="#main-content" className="skip-link">
        Skip to main content
      </a>

      {/* Semantic Shell Layout */}
      <Navbar />

      <main id="main-content">
        <Hero />
        
        <div id="about">
          <Mission />
        </div>
        
        <ProductShowcase />
        
        <IndustryCards />
        
        <Contact />
      </main>

      <footer 
        style={{
          borderTop: "var(--glass-border)",
          backgroundColor: "var(--bg-primary)",
          padding: "48px 24px",
          textAlign: "center",
          color: "var(--neutral)",
          fontSize: "14px"
        }}
      >
        <div style={{ maxWidth: "var(--content-max-width)", margin: "0 auto", display: "grid", gap: "16px" }}>
          <div>
            &copy; {new Date().getFullYear()} MediIntel. All rights reserved. Clinical support tools powered by MediIntel Intelligence.
          </div>
          <div style={{ display: "flex", justifyContent: "center", gap: "24px", fontSize: "13px" }}>
            <a href="#about" className="nav-link">About</a>
            <a href="#products" className="nav-link">Products</a>
            <a href="#contact" className="nav-link">Contact</a>
            <a href="/login" className="nav-link">Console Login</a>
          </div>
        </div>
      </footer>
    </>
  );
}
