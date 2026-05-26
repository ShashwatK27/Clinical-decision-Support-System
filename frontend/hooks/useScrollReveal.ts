import { RefObject, useEffect } from "react";

export default function useScrollReveal(containerRef?: RefObject<HTMLElement>) {
  useEffect(() => {
    // Check if the user prefers reduced motion
    const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (prefersReducedMotion) return;

    const observerOptions = {
      root: null,
      rootMargin: "0px 0px -80px 0px", // Trigger slightly before element enters full view
      threshold: 0.1, // Trigger when 10% visible
    };

    const revealCallback = (entries: IntersectionObserverEntry[], observer: IntersectionObserver) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("reveal-active");
          observer.unobserve(entry.target); // Stop tracking after it reveals once
        }
      });
    };

    const observer = new IntersectionObserver(revealCallback, observerOptions);
    
    // Select elements within the scoped ref container if provided, otherwise fallback to entire document
    const scope = containerRef?.current || document.documentElement;
    const elements = scope.querySelectorAll(".scroll-reveal");

    elements.forEach((el) => observer.observe(el));

    return () => {
      elements.forEach((el) => observer.unobserve(el));
    };
  }, [containerRef]);
}
