import { useState, useEffect } from "react";

export default function useNavbarScroll(threshold: number = 20): boolean {
  const [isScrolled, setIsScrolled] = useState<boolean>(false);

  useEffect(() => {
    const handleScroll = () => {
      if (window.scrollY > threshold) {
        setIsScrolled(true);
      } else {
        setIsScrolled(false);
      }
    };

    // Attach passive listener to optimize scroll performance (no scroll blocking)
    window.addEventListener("scroll", handleScroll, { passive: true });
    
    // Run once on mount in case user loads page scrolled down
    handleScroll();

    return () => {
      window.removeEventListener("scroll", handleScroll);
    };
  }, [threshold]);

  return isScrolled;
}
