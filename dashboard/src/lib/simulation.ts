/* simulation.ts – Hybrid Monte Carlo + ML collision engine */

export const PROTON_MASS = 0.938272088;

export interface SimConfig {
  numEvents: number;
  energyMin: number;
  energyMax: number;
  thetaMin: number;
  thetaMax: number;
  particleMass: number;
  seed?: number;

  useML?: boolean;
}

export const DEFAULT_CONFIG: SimConfig = {
  numEvents: 200,
  energyMin: 10,
  energyMax: 100,
  thetaMin: 0,
  thetaMax: Math.PI,
  particleMass: PROTON_MASS,

  useML: true
};

export interface CollisionEvent {
  id: number;
  energyA: number;
  energyB: number;
  theta: number;
  phi: number;
  px: number;
  py: number;
  pz: number;
  finalEnergy1: number;
  finalEnergy2: number;
  sqrtS: number;
}


/* ================= RNG ================= */

function splitmix32(a: number) {

  return () => {

    a |= 0;
    a = (a + 0x9e3779b9) | 0;

    let t = a ^ (a >>> 16);

    t = Math.imul(t, 0x21f0aaad);
    t ^= t >>> 15;

    t = Math.imul(t, 0x735a2d97);
    t ^= t >>> 15;

    return (t >>> 0) / 4294967296;
  };

}


/* ================= MONTE CARLO ================= */

function generateKinematics(cfg: SimConfig) {

  const rng =
    cfg.seed !== undefined ? splitmix32(cfg.seed) : () => Math.random();

  const cosThetaMin = Math.cos(cfg.thetaMax);
  const cosThetaMax = Math.cos(cfg.thetaMin);

  const mass2 = cfg.particleMass * cfg.particleMass;

  const inputs = [];

  for (let i = 0; i < cfg.numEvents; i++) {

    const eA = cfg.energyMin + rng() * (cfg.energyMax - cfg.energyMin);
    const eB = cfg.energyMin + rng() * (cfg.energyMax - cfg.energyMin);

    const cosTheta =
      cosThetaMin + rng() * (cosThetaMax - cosThetaMin);

    const theta = Math.acos(cosTheta);

    const sinTheta = Math.sin(theta);

    const phiA = rng() * 2 * Math.PI;
    const phiB = rng() * 2 * Math.PI;

    const pA = Math.sqrt(Math.max(eA * eA - mass2, 0));
    const pB = Math.sqrt(Math.max(eB * eB - mass2, 0));

    const pxA = pA * sinTheta * Math.cos(phiA);
    const pyA = pA * sinTheta * Math.sin(phiA);
    const pzA = pA * cosTheta;

    const pxB = pB * sinTheta * Math.cos(phiB);
    const pyB = pB * sinTheta * Math.sin(phiB);
    const pzB = -pB * cosTheta;

    const px = pxA + pxB;
    const py = pyA + pyB;
    const pz = pzA + pzB;

    inputs.push({

      id: i,

      energyA: eA,
      energyB: eB,

      theta,
      phi: phiA,

      px,
      py,
      pz

    });

  }

  return inputs;

}


/* ================= PURE MONTE CARLO ================= */

function runMonteCarlo(cfg: SimConfig): CollisionEvent[] {

  const kin = generateKinematics(cfg);

  return kin.map(k => {

    const totalE = k.energyA + k.energyB;

    const asymmetry = 0.5 + (Math.random() - 0.5) * 0.2;

    const e1 = totalE * asymmetry;

    const e2 = totalE - e1;

    const p2 = k.px * k.px + k.py * k.py + k.pz * k.pz;

    const s = totalE * totalE - p2;

    return {

      ...k,

      finalEnergy1: e1,
      finalEnergy2: e2,

      sqrtS: Math.sqrt(Math.max(s, 0))

    };

  });

}


/* ================= ML BATCH ================= */

async function runML(cfg: SimConfig): Promise<CollisionEvent[]> {

  const kin = generateKinematics(cfg);

  const payload = kin.map(k => ({

    particle_A_energy: k.energyA,
    particle_B_energy: k.energyB,
    collision_angle: k.theta,

    momentum_x: k.px,
    momentum_y: k.py,
    momentum_z: k.pz

  }));


  const res = await fetch(

    "http://127.0.0.1:8000/predict_batch",

    {

      method: "POST",

      headers: {
        "Content-Type": "application/json"
      },

      body: JSON.stringify(payload)

    }

  );


  const preds = await res.json();


  return kin.map((k, i) => ({

    ...k,

    finalEnergy1: preds[i].final_energy_1,
    finalEnergy2: preds[i].final_energy_2,

    sqrtS: preds[i].sqrt_s

  }));

}


/* ================= MAIN ================= */

export async function runSimulation(

  cfg: SimConfig

): Promise<CollisionEvent[]> {

  if (cfg.useML)
    return await runML(cfg);

  return runMonteCarlo(cfg);

}


/* ================= ACCURACY ================= */

export async function compareMLvsMC(

  cfg: SimConfig

) {

  /* identical random sequence */

  const baseCfg = {

    ...cfg,

    seed: cfg.seed ?? 42

  };


  const mc = runMonteCarlo(baseCfg);

  const ml = await runML(baseCfg);


  const errors = mc.map(

    (m, i) => Math.abs(m.sqrtS - ml[i].sqrtS)

  );


  const mae =
    errors.reduce((a, b) => a + b, 0) / errors.length;


  const meanS =
    mc.reduce((a, b) => a + b.sqrtS, 0) / mc.length;


  return {

    mae,

    relError: mae / meanS

  };

}


/* ================= STATS ================= */

export function mean(arr: number[]) {

  return arr.reduce((a, b) => a + b, 0) / arr.length;

}

export function std(arr: number[]) {

  const m = mean(arr);

  return Math.sqrt(

    arr.reduce((s, x) => s + (x - m) ** 2, 0) / arr.length

  );

}

export function percentile(arr: number[], p: number) {

  const s = [...arr].sort((a, b) => a - b);

  const idx = (p / 100) * (s.length - 1);

  const lo = Math.floor(idx);

  const hi = Math.ceil(idx);

  return s[lo] + (s[hi] - s[lo]) * (idx - lo);

}


export interface HistBin {

  x: number;
  y: number;

}

export function histogram(data: number[], bins = 60) {

  const mn = Math.min(...data);

  const mx = Math.max(...data);

  const w = (mx - mn) / bins || 1;

  const c = new Array(bins).fill(0);

  for (const v of data) {

    const i = Math.min(
      Math.floor((v - mn) / w),
      bins - 1
    );

    c[i]++;

  }

  return c.map((v, i) => ({

    x: mn + (i + 0.5) * w,
    y: v

  }));

}


export function correlation(a: number[], b: number[]) {

  const ma = mean(a);

  const mb = mean(b);

  let num = 0, da = 0, db = 0;

  for (let i = 0; i < a.length; i++) {

    const x = a[i] - ma;

    const y = b[i] - mb;

    num += x * y;

    da += x * x;

    db += y * y;

  }

  return num / Math.sqrt(da * db);

}

export function detectAnomalies(

  events: CollisionEvent[],
  percentileCutoff = 95

) {

  const values = events.map(e => e.sqrtS);

  const threshold = percentile(values, percentileCutoff);

  return events.map(e => ({

    x: e.finalEnergy1,
    y: e.finalEnergy2,

    anomaly: e.sqrtS >= threshold

  }));

}