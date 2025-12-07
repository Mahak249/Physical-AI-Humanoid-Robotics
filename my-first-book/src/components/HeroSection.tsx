import React from 'react';
import Link from '@docusaurus/Link';
import styles from './HeroSection.module.css';

export default function HeroSection(): JSX.Element {
  return (
    <section className={styles.hero}>
      <div className={styles.heroContainer}>
        <div className={styles.heroContent}>
          <h1 className={styles.heroTitle}>
            Physical AI &<br />
            Humanoid Robotics
          </h1>
          <p className={styles.heroSubtitle}>
            Master the future of AI-powered robotics with hands-on learning.
            Explore ROS 2, Isaac Sim, VLA models, and real-world humanoid platforms.
          </p>
          <div className={styles.heroButtons}>
            <Link
              className={styles.primaryButton}
              to="/docs/intro">
              Start Learning →
            </Link>
            <Link
              className={styles.secondaryButton}
              to="/docs/chapter-01-intro">
              View Chapters
            </Link>
          </div>

          {/* Feature Pills */}
          <div className={styles.featurePills}>
            <div className={styles.pill}>
              <span className={styles.pillIcon}>🤖</span>
              <span>6 Comprehensive Chapters</span>
            </div>
            <div className={styles.pill}>
              <span className={styles.pillIcon}>⚡</span>
              <span>Hands-on Code Examples</span>
            </div>
            <div className={styles.pill}>
              <span className={styles.pillIcon}>🎯</span>
              <span>Industry Best Practices</span>
            </div>
          </div>
        </div>

        <div className={styles.heroVisual}>
          <div className={styles.robotFrame}>
            <div className={styles.glowEffect}></div>
            {/* SVG Robot Illustration */}
            <svg
              className={styles.robotSvg}
              viewBox="0 0 400 500"
              fill="none"
              xmlns="http://www.w3.org/2000/svg">

              {/* Robot Head */}
              <g className={styles.robotHead}>
                <rect x="150" y="80" width="100" height="80" rx="10" fill="url(#headGradient)" />
                <circle cx="175" cy="110" r="8" fill="#3b82f6" className={styles.eye} />
                <circle cx="225" cy="110" r="8" fill="#3b82f6" className={styles.eye} />
                <rect x="170" y="135" width="60" height="3" rx="1.5" fill="#a855f7" opacity="0.6" />
                <rect x="170" y="145" width="40" height="3" rx="1.5" fill="#a855f7" opacity="0.4" />
              </g>

              {/* Robot Body */}
              <g className={styles.robotBody}>
                <rect x="140" y="170" width="120" height="140" rx="15" fill="url(#bodyGradient)" />
                <circle cx="200" cy="240" r="25" fill="#a855f7" opacity="0.2" />
                <circle cx="200" cy="240" r="15" fill="#a855f7" opacity="0.4" />

                {/* AI Core Lines */}
                <line x1="160" y1="200" x2="240" y2="200" stroke="#3b82f6" strokeWidth="2" opacity="0.5" />
                <line x1="160" y1="210" x2="240" y2="210" stroke="#3b82f6" strokeWidth="2" opacity="0.5" />
                <line x1="160" y1="220" x2="240" y2="220" stroke="#3b82f6" strokeWidth="2" opacity="0.5" />
              </g>

              {/* Robot Arms */}
              <g className={styles.robotArms}>
                {/* Left Arm */}
                <rect x="100" y="180" width="35" height="100" rx="8" fill="url(#armGradient)" />
                <circle cx="117" cy="290" r="15" fill="#a855f7" opacity="0.3" />

                {/* Right Arm */}
                <rect x="265" y="180" width="35" height="100" rx="8" fill="url(#armGradient)" />
                <circle cx="283" cy="290" r="15" fill="#a855f7" opacity="0.3" />
              </g>

              {/* Robot Legs */}
              <g className={styles.robotLegs}>
                {/* Left Leg */}
                <rect x="160" y="315" width="35" height="120" rx="8" fill="url(#legGradient)" />
                <rect x="155" y="435" width="45" height="20" rx="5" fill="#1e293b" />

                {/* Right Leg */}
                <rect x="205" y="315" width="35" height="120" rx="8" fill="url(#legGradient)" />
                <rect x="200" y="435" width="45" height="20" rx="5" fill="#1e293b" />
              </g>

              {/* Holographic Tablet */}
              <g className={styles.holographicTablet}>
                <rect x="280" y="200" width="80" height="100" rx="5" fill="#0F1629" opacity="0.6" stroke="#a855f7" strokeWidth="2" />
                <rect x="290" y="210" width="60" height="3" rx="1.5" fill="#3b82f6" opacity="0.8" />
                <rect x="290" y="220" width="50" height="3" rx="1.5" fill="#3b82f6" opacity="0.6" />
                <rect x="290" y="230" width="55" height="3" rx="1.5" fill="#3b82f6" opacity="0.7" />
                <rect x="290" y="250" width="60" height="30" rx="3" fill="#a855f7" opacity="0.2" />
              </g>

              {/* Gradients */}
              <defs>
                <linearGradient id="headGradient" x1="150" y1="80" x2="250" y2="160">
                  <stop offset="0%" stopColor="#1e293b" />
                  <stop offset="100%" stopColor="#0f172a" />
                </linearGradient>
                <linearGradient id="bodyGradient" x1="140" y1="170" x2="260" y2="310">
                  <stop offset="0%" stopColor="#1e293b" />
                  <stop offset="100%" stopColor="#0f172a" />
                </linearGradient>
                <linearGradient id="armGradient" x1="0" y1="180" x2="0" y2="280">
                  <stop offset="0%" stopColor="#1e293b" />
                  <stop offset="100%" stopColor="#0f172a" />
                </linearGradient>
                <linearGradient id="legGradient" x1="0" y1="315" x2="0" y2="435">
                  <stop offset="0%" stopColor="#1e293b" />
                  <stop offset="100%" stopColor="#0f172a" />
                </linearGradient>
              </defs>
            </svg>
          </div>
        </div>
      </div>
    </section>
  );
}
