import sharp from 'sharp';
import fs from 'fs';
import path from 'path';

const demoDir = path.resolve('public/demo');
if (!fs.existsSync(demoDir)) {
  fs.mkdirSync(demoDir, { recursive: true });
}

// 1. Femur with cortical bone and focal lucency
const femurSvg = `
<svg width="600" height="600" viewBox="0 0 600 600" xmlns="http://www.w3.org/2000/svg">
  <rect width="600" height="600" fill="#080c14" />
  <!-- Soft tissue silhouette -->
  <path d="M 200,20 C 180,200 170,400 190,580 L 410,580 C 430,400 420,200 400,20 Z" fill="#141c2b" />
  
  <!-- Femur bone shaft -->
  <path d="M 240,40 C 245,200 248,350 250,560 L 350,560 C 352,350 355,200 360,40 Z" fill="#8899ac" opacity="0.9" />
  <!-- Medullary canal (marrow) -->
  <path d="M 270,40 C 272,200 274,350 275,560 L 325,560 C 326,350 328,200 330,40 Z" fill="#4d5c6f" />
  
  <!-- Focal cortical thinning / lesion zone (demo metastasis) -->
  <ellipse cx="255" cy="280" rx="22" ry="34" fill="#1c2533" opacity="0.9" />
  <circle cx="260" cy="275" r="14" fill="#0c111a" />
  <ellipse cx="340" cy="380" rx="18" ry="24" fill="#1c2533" opacity="0.8" />
  
  <!-- Caliper / text overlay simulated on X-ray -->
  <text x="30" y="50" fill="#475569" font-family="monospace" font-size="14">OS-XR-01 | FEMUR AP</text>
  <text x="30" y="70" fill="#334155" font-family="monospace" font-size="12">75kVp 12mAs LATERAL L</text>
  <text x="540" y="50" fill="#64748b" font-family="sans-serif" font-weight="bold" font-size="24">L</text>
</svg>
`;

// 2. Pelvis / Hip joint with cortical thickening (demo Paget)
const pelvisSvg = `
<svg width="600" height="600" viewBox="0 0 600 600" xmlns="http://www.w3.org/2000/svg">
  <rect width="600" height="600" fill="#080c14" />
  <path d="M 120,40 C 220,90 380,90 480,40 L 520,380 C 440,480 360,540 300,560 C 240,540 160,480 80,380 Z" fill="#141c2b" />
  <!-- Iliac wing & acetabulum -->
  <path d="M 160,80 C 230,120 370,120 440,80 C 470,220 430,340 380,370 C 340,320 260,320 220,370 C 170,340 130,220 160,80 Z" fill="#94a3b8" />
  <!-- Sclerotic expansion zone -->
  <circle cx="370" cy="320" r="50" fill="#e2e8f0" opacity="0.85" />
  <ellipse cx="370" cy="330" rx="35" ry="30" fill="#cbd5e1" />
  <text x="30" y="50" fill="#475569" font-family="monospace" font-size="14">OS-XR-02 | PELVIS AP</text>
  <text x="30" y="70" fill="#334155" font-family="monospace" font-size="12">80kVp 16mAs RIGHT HIP</text>
  <text x="540" y="50" fill="#64748b" font-family="sans-serif" font-weight="bold" font-size="24">R</text>
</svg>
`;

// 3. Tibia healthy normal bone
const tibiaSvg = `
<svg width="600" height="600" viewBox="0 0 600 600" xmlns="http://www.w3.org/2000/svg">
  <rect width="600" height="600" fill="#080c14" />
  <path d="M 210,30 C 200,200 195,380 210,570 L 390,570 C 405,380 400,200 390,30 Z" fill="#141c2b" />
  <!-- Intact tibial shaft -->
  <path d="M 250,50 L 260,550 L 340,550 L 350,50 Z" fill="#94a3b8" />
  <path d="M 275,50 L 280,550 L 320,550 L 325,50 Z" fill="#475569" />
  <!-- Fine trabecular matrix lines -->
  <line x1="265" y1="180" x2="335" y2="180" stroke="#cbd5e1" stroke-width="1.5" opacity="0.6" />
  <line x1="268" y1="260" x2="332" y2="260" stroke="#cbd5e1" stroke-width="1.5" opacity="0.6" />
  <line x1="270" y1="340" x2="330" y2="340" stroke="#cbd5e1" stroke-width="1.5" opacity="0.6" />
  <text x="30" y="50" fill="#475569" font-family="monospace" font-size="14">OS-XR-03 | TIBIA/FIBULA</text>
  <text x="30" y="70" fill="#334155" font-family="monospace" font-size="12">65kVp 8mAs HEALTHY ADULT</text>
  <text x="540" y="50" fill="#64748b" font-family="sans-serif" font-weight="bold" font-size="24">L</text>
</svg>
`;

async function generate() {
  await sharp(Buffer.from(femurSvg)).png().toFile(path.join(demoDir, 'femur-xray-1.png'));
  await sharp(Buffer.from(pelvisSvg)).png().toFile(path.join(demoDir, 'pelvis-xray-2.png'));
  await sharp(Buffer.from(tibiaSvg)).png().toFile(path.join(demoDir, 'tibia-xray-3.png'));
  console.log('Demo X-ray assets generated successfully in public/demo');
}

generate().catch(console.error);
