import { useState } from "react";
import Link from "next/link";
import useNavbarScroll from "../../hooks/useNavbarScroll";

export default function Navbar() {
  const isScrolled = useNavbarScroll(20);
  const [isOpen, setIsOpen] = useState<boolean>(false);

  const toggleMenu = () => {
    setIsOpen(!isOpen);
  };

  return (
    <header 
      className={`navbar-container ${isScrolled ? "scrolled" : ""}`}
      style={{
        position: "fixed",
        top: 0,
        left: 0,
        width: "100%",
        height: "80px",
        zIndex: 1000,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        transition: "background-color 0.3s ease, border-color 0.3s ease, backdrop-filter 0.3s ease",
        backgroundColor: isScrolled ? "rgba(18, 18, 20, 0.85)" : "transparent",
        backdropFilter: isScrolled ? "var(--glass-blur)" : "none",
        borderBottom: isScrolled ? "var(--glass-border)" : "1px solid transparent",
        padding: "0 24px"
      }}
    >
      <div 
        style={{
          width: "100%",
          maxWidth: "var(--content-max-width)",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between"
        }}
      >
        {/* Brand Logo */}
        <Link 
          href="/" 
          style={{ 
            fontSize: "24px", 
            fontWeight: 800, 
            color: "var(--text-primary)", 
            display: "flex", 
            alignItems: "center",
            gap: "8px",
            letterSpacing: "var(--letter-spacing-tight)"
          }}
          aria-label="CDSS AI Home"
        >
          <span style={{ color: "var(--accent-primary)" }}>CDSS</span> AI
        </Link>

        {/* Desktop Navigation Links */}
        <nav 
          style={{ display: "none" }} 
          className="desktop-nav"
          aria-label="Primary Navigation"
        >
          <ul style={{ display: "flex", gap: "32px", listStyle: "none", alignItems: "center" }}>
            <li>
              <a href="#about" className="nav-link">About</a>
            </li>
            <li>
              <a href="#modules" className="nav-link">Modules</a>
            </li>
            <li>
              <a href="#specialties" className="nav-link">Specialties</a>
            </li>
            <li>
              <a href="#contact" className="nav-link">Contact</a>
            </li>
          </ul>
        </nav>

        {/* Desktop CTAs */}
        <div style={{ display: "none", gap: "16px", alignItems: "center" }} className="desktop-ctas">
          <Link href="/login" style={{ color: "var(--text-primary)", fontWeight: 500 }} className="nav-link">
            Sign In
          </Link>
          <Link 
            href="/dashboard" 
            className="btn-hover"
            style={{ 
              background: "var(--white)", 
              color: "var(--black)", 
              padding: "10px 22px", 
              borderRadius: "999px",
              fontWeight: 700,
              fontSize: "14px",
              boxShadow: "0 10px 20px rgba(255,255,255,0.05)"
            }}
          >
            Launch Console
          </Link>
        </div>

        {/* Mobile Navigation Toggle */}
        <button 
          onClick={toggleMenu}
          aria-expanded={isOpen}
          aria-controls="mobile-nav-menu"
          aria-label="Toggle navigation menu"
          style={{
            background: "transparent",
            border: "none",
            cursor: "pointer",
            width: "32px",
            height: "32px",
            display: "flex",
            flexDirection: "column",
            justifyContent: "space-around",
            padding: "6px",
            zIndex: 1001
          }}
          className="hamburger-btn"
        >
          <span style={{
            width: "100%",
            height: "2px",
            backgroundColor: "var(--text-primary)",
            transition: "transform 0.3s ease, opacity 0.3s ease",
            transform: isOpen ? "rotate(45deg) translate(5px, 5px)" : "none"
          }} />
          <span style={{
            width: "100%",
            height: "2px",
            backgroundColor: "var(--text-primary)",
            transition: "opacity 0.3s ease",
            opacity: isOpen ? 0 : 1
          }} />
          <span style={{
            width: "100%",
            height: "2px",
            backgroundColor: "var(--text-primary)",
            transition: "transform 0.3s ease, opacity 0.3s ease",
            transform: isOpen ? "rotate(-45deg) translate(5px, -5px)" : "none"
          }} />
        </button>
      </div>

      {/* Mobile Fullscreen Navigation Overlay */}
      <div
        id="mobile-nav-menu"
        style={{
          position: "fixed",
          top: 0,
          right: 0,
          width: "100%",
          height: "100vh",
          backgroundColor: "rgba(18, 18, 20, 0.98)",
          backdropFilter: "blur(20px)",
          zIndex: 999,
          display: isOpen ? "flex" : "none",
          flexDirection: "column",
          justifyContent: "center",
          alignItems: "center",
          gap: "40px",
          transition: "opacity 0.3s ease"
        }}
      >
        <ul style={{ listStyle: "none", textAlign: "center", display: "grid", gap: "24px" }}>
          <li>
            <a href="#about" onClick={toggleMenu} style={{ fontSize: "24px", fontWeight: 600 }}>About</a>
          </li>
          <li>
            <a href="#modules" onClick={toggleMenu} style={{ fontSize: "24px", fontWeight: 600 }}>Modules</a>
          </li>
          <li>
            <a href="#specialties" onClick={toggleMenu} style={{ fontSize: "24px", fontWeight: 600 }}>Specialties</a>
          </li>
          <li>
            <a href="#contact" onClick={toggleMenu} style={{ fontSize: "24px", fontWeight: 600 }}>Contact</a>
          </li>
        </ul>

        <div style={{ display: "grid", gap: "16px", width: "80%", maxWidth: "320px", marginTop: "20px" }}>
          <Link 
            href="/login" 
            onClick={toggleMenu} 
            style={{ 
              color: "var(--text-primary)", 
              border: "var(--glass-border)", 
              padding: "14px", 
              borderRadius: "999px",
              textAlign: "center",
              fontWeight: 600,
              background: "rgba(255, 255, 255, 0.02)"
            }}
          >
            Sign In
          </Link>
          <Link 
            href="/dashboard" 
            onClick={toggleMenu}
            style={{ 
              background: "var(--white)", 
              color: "var(--black)", 
              padding: "14px", 
              borderRadius: "999px",
              textAlign: "center",
              fontWeight: 700
            }}
          >
            Launch Console
          </Link>
        </div>
      </div>

      <style jsx global>{`
        @media (min-width: 769px) {
          .desktop-nav {
            display: block !important;
          }
          .desktop-ctas {
            display: flex !important;
          }
          .hamburger-btn {
            display: none !important;
          }
        }
        .nav-link {
          color: var(--neutral);
          font-weight: 500;
          font-size: 15px;
          transition: color 0.2s ease, text-shadow 0.2s ease;
        }
        .nav-link:hover {
          color: var(--text-primary);
          text-shadow: 0 0 10px rgba(255, 255, 255, 0.3);
        }
      `}</style>
    </header>
  );
}
