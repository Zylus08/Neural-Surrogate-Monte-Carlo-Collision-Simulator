"use client";

import { useState, useCallback } from "react";
import { motion } from "framer-motion";
import {
  runSimulation,
  compareMLvsMC,
  DEFAULT_CONFIG,
  mean,
  std,
  histogram,
  correlation,
  detectAnomalies,
  type SimConfig,
  type CollisionEvent,
} from "@/lib/simulation";

import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ScatterChart,
  Scatter,
} from "recharts";

import GlowCard from "@/components/GlowCard";


export default function SimulationPage() {

  const [useML, setUseML] = useState(true);

  const [config, setConfig] = useState<SimConfig>({
    ...DEFAULT_CONFIG,
    numEvents: 1000
  });

  const [events, setEvents] = useState<CollisionEvent[]>([]);

  const [stats, setStats] = useState<any>(null);

  const [modelError, setModelError] = useState<any>(null);

  const [running, setRunning] = useState(false);



  const handleRun = useCallback(async () => {

    setRunning(true);

    setEvents([]);

    setStats(null);

    setModelError(null);


    const t0 = performance.now();


    const result = await runSimulation({

      ...config,
      useML

    });


    const t1 = performance.now();


    const sqrtSArr = result.map(e => e.sqrtS);

    const e1Arr = result.map(e => e.finalEnergy1);

    const e2Arr = result.map(e => e.finalEnergy2);


    setEvents(result);


    setStats({

      meanSqrtS: mean(sqrtSArr),

      stdSqrtS: std(sqrtSArr),

      corr: correlation(e1Arr, e2Arr),

      time: (t1 - t0) / 1000

    });


    if (useML) {

      const err = await compareMLvsMC({

        ...config,
        numEvents: 200

      });

      setModelError(err);

    }


    setRunning(false);

  }, [config, useML]);



  const hist = histogram(

    events.map(e => e.sqrtS),

    50

  );


  const scatter = detectAnomalies(

    events.slice(0, 2000)

  );



  return (

    <div className="max-w-6xl mx-auto px-6 py-12 space-y-8">


      {/* TITLE */}

      <h1 className="text-4xl font-bold text-center">

        Particle Collision Simulator

      </h1>



      {/* CONTROL PANEL */}

      <div className="glass p-6 space-y-6">


        {/* MODE */}

        <div>

          <label className="block text-sm mb-2">

            Simulation Mode

          </label>


          <div className="flex gap-6">

            <label className="flex gap-2 items-center">

              <input

                type="radio"

                checked={useML}

                onChange={() => setUseML(true)}

              />

              AI Accelerator

            </label>


            <label className="flex gap-2 items-center">

              <input

                type="radio"

                checked={!useML}

                onChange={() => setUseML(false)}

              />

              Monte Carlo

            </label>

          </div>

        </div>



        {/* EVENTS */}

        <div>

          <label className="block text-sm mb-2">

            Number of Events

          </label>


          <input

            type="range"

            min={100}

            max={5000}

            step={100}

            value={config.numEvents}

            onChange={e =>

              setConfig({

                ...config,
                numEvents: Number(e.target.value)

              })

            }

          />

          <div className="text-cyan-400">

            {config.numEvents}

          </div>

        </div>



        {/* ENERGY MIN */}

        <div>

          <label className="block text-sm mb-2">

            Min Beam Energy

          </label>


          <input

            type="range"

            min={1}

            max={config.energyMax - 1}

            value={config.energyMin}

            onChange={e =>

              setConfig({

                ...config,
                energyMin: Number(e.target.value)

              })

            }

          />

          {config.energyMin}

        </div>



        {/* ENERGY MAX */}

        <div>

          <label className="block text-sm mb-2">

            Max Beam Energy

          </label>


          <input

            type="range"

            min={config.energyMin + 1}

            max={1000}

            value={config.energyMax}

            onChange={e =>

              setConfig({

                ...config,
                energyMax: Number(e.target.value)

              })

            }

          />

          {config.energyMax}

        </div>



        {/* THETA */}

        <div>

          <label className="block text-sm mb-2">

            Scattering Angle θ

          </label>


          <input

            type="range"

            min={0.1}

            max={3.14}

            step={0.01}

            value={config.thetaMax}

            onChange={e =>

              setConfig({

                ...config,
                thetaMax: Number(e.target.value)

              })

            }

          />

          {config.thetaMax.toFixed(2)}

        </div>



        <button

          onClick={handleRun}

          disabled={running}

          className="px-6 py-3 bg-cyan-500 rounded-lg"

        >

          {running ? "Running Simulation..." : "Run Simulation"}

        </button>


      </div>



      {/* STATS */}

      {stats && (

        <div className="grid grid-cols-2 md:grid-cols-5 gap-4">


          <GlowCard glowColor="cyan">

            Events

            <div>

              {events.length}

            </div>

          </GlowCard>


          <GlowCard glowColor="purple">

            Mean √s

            <div>

              {stats.meanSqrtS.toFixed(2)}

            </div>

          </GlowCard>


          <GlowCard glowColor="blue">

            Std Dev

            <div>

              {stats.stdSqrtS.toFixed(2)}

            </div>

          </GlowCard>


          <GlowCard glowColor="green">

            Runtime

            <div>

              {stats.time.toFixed(3)} s

            </div>

          </GlowCard>


          {modelError && (

            <GlowCard glowColor="pink">

              Model Error

              <div>

                {modelError.mae.toFixed(2)}

              </div>

              <div>

                {(modelError.relError * 100).toFixed(2)} %

              </div>

            </GlowCard>

          )}


        </div>

      )}



      {/* HISTOGRAM */}

      <div className="glass p-6">

        <h2 className="mb-4">

          Invariant Mass Distribution

        </h2>


        <ResponsiveContainer width="100%" height={300}>

          <AreaChart data={hist}>

            <CartesianGrid strokeDasharray="3 3" />

            <XAxis dataKey="x" />

            <YAxis />

            <Tooltip />

            <Area dataKey="y" />

          </AreaChart>

        </ResponsiveContainer>

      </div>



      {/* SCATTER */}

      <div className="glass p-6">

        <h2 className="mb-4">

          Energy Correlation

        </h2>


        <ResponsiveContainer width="100%" height={300}>

          <ScatterChart>

            <CartesianGrid strokeDasharray="3 3" />

            <XAxis dataKey="x" />

            <YAxis dataKey="y" />

            <Tooltip />

            <Scatter

              data={scatter.filter(p => !p.anomaly)}

              fill="#00eaff"

            />

            <Scatter

              data={scatter.filter(p => p.anomaly)}

              fill="#ff2e63"

            />

          </ScatterChart>

        </ResponsiveContainer>

      </div>



    </div>

  );

}