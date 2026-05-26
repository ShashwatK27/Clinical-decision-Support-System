import { useRef } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, Html } from "@react-three/drei";
import * as THREE from "three";

interface NetworkSceneProps {
  drugs: string[];
  interactions: Array<{ drugs: string[]; severity: string; effect: string }>;
  onSelectInteraction: (drugPair: string[], severity: string, effect: string) => void;
}

// Symmetrical pre-allocated coordinates to ensure clean composition
const NODE_COORDINATES: Array<[number, number, number]> = [
  [0, 0.8, 0],         // Node 1 (Center Top)
  [-0.9, 0.1, 0.4],     // Node 2 (Left Mid)
  [0.9, 0.1, -0.4],    // Node 3 (Right Mid)
  [-0.6, -0.6, -0.3],  // Node 4 (Left Bottom)
  [0.6, -0.6, 0.3],    // Node 5 (Right Bottom)
  [0, -0.1, -0.8],     // Node 6 (Back Deep)
];

// Glowing Neon Connecting line between drug nodes
function ConnectionLine({ 
  start, 
  end, 
  isWarning, 
  onClick 
}: { 
  start: [number, number, number]; 
  end: [number, number, number]; 
  isWarning: boolean;
  onClick?: () => void;
}) {
  const lineRef = useRef<THREE.Mesh>(null);

  // Animate pulse/vibration for warning connections
  useFrame((state) => {
    if (!lineRef.current) return;
    const t = state.clock.getElapsedTime();
    if (isWarning) {
      // Fast warning vibration + scale pulse
      const pulse = 1.0 + Math.sin(t * 12) * 0.12;
      lineRef.current.scale.set(pulse, 1.0, pulse);
    } else {
      // Soft wave breathe
      const breathe = 1.0 + Math.sin(t * 2) * 0.05;
      lineRef.current.scale.set(breathe, 1.0, breathe);
    }
  });

  // Math to align a cylinder between two 3D vectors
  const pStart = new THREE.Vector3(...start);
  const pEnd = new THREE.Vector3(...end);
  const distance = pStart.distanceTo(pEnd);
  const position = pStart.clone().add(pEnd).multiplyScalar(0.5);
  
  // Calculate orientation
  const direction = pEnd.clone().sub(pStart).normalize();
  const up = new THREE.Vector3(0, 1, 0);
  const quaternion = new THREE.Quaternion().setFromUnitVectors(up, direction);

  return (
    <mesh 
      ref={lineRef} 
      position={position} 
      quaternion={quaternion} 
      onClick={onClick}
    >
      <cylinderGeometry args={[isWarning ? 0.045 : 0.015, isWarning ? 0.045 : 0.015, distance, 8]} />
      <meshBasicMaterial 
        color={isWarning ? "#ef4444" : "#10b981"} 
        transparent 
        opacity={isWarning ? 0.95 : 0.45} 
      />
    </mesh>
  );
}

// Float floating animation for entire group of drug spheres
function FloatingGroup({ children }: { children: React.ReactNode }) {
  const groupRef = useRef<THREE.Group>(null);
  
  useFrame((state) => {
    const t = state.clock.getElapsedTime();
    if (groupRef.current) {
      // Gentle spatial sway
      groupRef.current.position.y = Math.sin(t * 0.8) * 0.05;
      groupRef.current.rotation.y = Math.cos(t * 0.3) * 0.02;
    }
  });

  return <group ref={groupRef}>{children}</group>;
}

export default function NetworkScene({ drugs, interactions, onSelectInteraction }: NetworkSceneProps) {
  // Map parsed drugs to symmetrical positions
  const mappedNodes = drugs.map((drug, index) => {
    const coords = NODE_COORDINATES[index % NODE_COORDINATES.length];
    
    // Check if this drug is involved in any severe interaction
    const matchedSevereInteraction = interactions.find(
      ix => ix.severity === "severe" && ix.drugs.includes(drug)
    );
    const matchedModerateInteraction = interactions.find(
      ix => ix.severity === "moderate" && ix.drugs.includes(drug)
    );

    let nodeType = "safe";
    if (matchedSevereInteraction) nodeType = "severe";
    else if (matchedModerateInteraction) nodeType = "moderate";

    return {
      name: drug.toUpperCase(),
      rawName: drug,
      coords,
      type: nodeType,
      interactionInfo: matchedSevereInteraction || matchedModerateInteraction
    };
  });

  // Calculate lines between drug nodes
  const linesToRender: Array<{
    id: string;
    start: [number, number, number];
    end: [number, number, number];
    isWarning: boolean;
    onClick: () => void;
  }> = [];

  for (let i = 0; i < mappedNodes.length; i++) {
    for (let j = i + 1; j < mappedNodes.length; j++) {
      const drugA = mappedNodes[i].rawName;
      const drugB = mappedNodes[j].rawName;

      // Find if an interaction covers this specific pair
      const activeIx = interactions.find(
        ix => ix.drugs.includes(drugA) && ix.drugs.includes(drugB)
      );

      const isWarning = activeIx ? activeIx.severity === "severe" : false;
      const clickHandler = () => {
        if (activeIx) {
          onSelectInteraction(activeIx.drugs, activeIx.severity, activeIx.effect);
        }
      };

      linesToRender.push({
        id: `${drugA}-${drugB}`,
        start: mappedNodes[i].coords,
        end: mappedNodes[j].coords,
        isWarning,
        onClick: clickHandler
      });
    }
  }

  return (
    <div style={{ width: "100%", height: "100%", minHeight: 400, position: "relative" }}>
      <Canvas 
        camera={{ position: [0, 0.2, 4.0], fov: 45 }} 
        gl={{ antialias: true, alpha: true }}
      >
        <ambientLight intensity={0.2} />
        <directionalLight position={[3, 5, 2]} intensity={1.5} />
        <directionalLight position={[-3, -3, -2]} intensity={0.4} color="#3b82f6" />
        
        {/* Soft neon floor reflection */}
        <pointLight position={[0, -1.8, 0]} intensity={0.6} color="#06b6d4" />

        {/* 1. Transparent Glass Cylinder Containment Jar */}
        <mesh position={[0, 0, 0]}>
          <cylinderGeometry args={[1.4, 1.4, 3.0, 32, 1, true]} />
          <meshPhysicalMaterial 
            color="#0a162e"
            roughness={0.05} 
            metalness={0.9}
            clearcoat={1.0}
            clearcoatRoughness={0.05}
            transmission={0.95} // High transmission for premium realistic glass
            ior={1.45}
            thickness={0.15}
            side={THREE.DoubleSide}
            transparent
            opacity={0.35}
          />
        </mesh>
        
        {/* Symmetrical metallic base and cap for glass container */}
        <mesh position={[0, -1.5, 0]}>
          <cylinderGeometry args={[1.46, 1.46, 0.08, 32]} />
          <meshStandardMaterial color="#0f172a" metalness={0.9} roughness={0.2} />
        </mesh>
        <mesh position={[0, 1.5, 0]}>
          <cylinderGeometry args={[1.46, 1.46, 0.08, 32]} />
          <meshStandardMaterial color="#0f172a" metalness={0.9} roughness={0.2} />
        </mesh>

        <FloatingGroup>
          {mappedNodes.length === 0 ? (
            // Holographic Status Panel if dome is empty
            <Html center transform position={[0, 0.2, 0]}>
              <div style={{
                background: "rgba(10, 16, 32, 0.8)",
                border: "1px solid rgba(6, 182, 212, 0.4)",
                padding: "16px 20px",
                borderRadius: 14,
                color: "#06b6d4",
                fontFamily: "'Outfit', sans-serif",
                textAlign: "center",
                width: 170,
                fontSize: 10,
                letterSpacing: 1.5,
                boxShadow: "0 0 20px rgba(6, 182, 212, 0.1)"
              }}>
                <div style={{ fontWeight: 800, marginBottom: 4 }}>DOME EMPTY</div>
                <div style={{ color: "#a1a1aa", fontSize: 8 }}>LOAD Rx SCAN TO VISUALIZE DRUG MATRIX</div>
              </div>
            </Html>
          ) : (
            <>
              {/* 2. Floating Drug Nodes */}
              {mappedNodes.map((node) => {
                const isSevere = node.type === "severe";
                const isModerate = node.type === "moderate";
                
                let sphereColor = "#10b981"; // Emerald safe
                let emissiveColor = "#047857";
                if (isSevere) {
                  sphereColor = "#ef4444"; // Crimson severe
                  emissiveColor = "#b91c1c";
                } else if (isModerate) {
                  sphereColor = "#f59e0b"; // Amber moderate
                  emissiveColor = "#b45309";
                }

                return (
                  <group key={node.name} position={node.coords}>
                    {/* Molecular Node Sphere */}
                    <mesh castShadow>
                      <sphereGeometry args={[0.22, 32, 32]} />
                      <meshStandardMaterial 
                        color={sphereColor} 
                        emissive={emissiveColor}
                        emissiveIntensity={0.6}
                        roughness={0.15}
                        metalness={0.5}
                      />
                    </mesh>

                    {/* Holographic Exclamation Warning Icon floating near severe node */}
                    {isSevere && (
                      <mesh position={[0.25, 0.25, 0.15]}>
                        <boxGeometry args={[0.07, 0.07, 0.07]} />
                        <meshBasicMaterial color="#ef4444" />
                      </mesh>
                    )}

                    {/* Floating Label for Medicine name */}
                    <Html center position={[0, 0.38, 0]} distanceFactor={3.6}>
                      <div style={{
                        padding: "4px 8px",
                        background: "rgba(10, 16, 32, 0.9)",
                        border: `1px solid ${isSevere ? "rgba(239, 68, 68, 0.5)" : isModerate ? "rgba(245, 158, 11, 0.5)" : "rgba(16, 185, 129, 0.3)"}`,
                        borderRadius: 6,
                        color: "#f8fafc",
                        fontSize: 9,
                        fontWeight: 700,
                        letterSpacing: 0.5,
                        whiteSpace: "nowrap",
                        boxShadow: `0 4px 10px rgba(0,0,0,0.5), 0 0 10px ${isSevere ? "rgba(239, 68, 68, 0.15)" : "transparent"}`,
                        display: "flex",
                        alignItems: "center",
                        gap: 4
                      }}>
                        {isSevere && <span style={{ color: "#ef4444" }}>⚠️</span>}
                        {node.name}
                      </div>
                    </Html>
                  </group>
                );
              })}

              {/* 3. Glowing Connecting Energy Pathways */}
              {linesToRender.map((line) => (
                <ConnectionLine 
                  key={line.id} 
                  start={line.start} 
                  end={line.end} 
                  isWarning={line.isWarning} 
                  onClick={line.onClick}
                />
              ))}
            </>
          )}
        </FloatingGroup>

        <OrbitControls 
          enableZoom={false} 
          maxPolarAngle={Math.PI / 1.7} 
          minPolarAngle={Math.PI / 2.4}
          maxAzimuthAngle={Math.PI / 4}
          minAzimuthAngle={-Math.PI / 4}
        />
      </Canvas>
    </div>
  );
}
