import type {ReactNode} from 'react';
import clsx from 'clsx';
import Heading from '@theme/Heading';
import Link from '@docusaurus/Link';
import styles from './styles.module.css';

type FeatureItem = {
  title: string;
  icon: string;
  description: ReactNode;
  link: string;
  chapter: string;
};

const FeatureList: FeatureItem[] = [
  {
    title: 'Introduction to Physical AI',
    icon: '🤖',
    chapter: 'Chapter 1',
    description: (
      <>
        Discover the fundamentals of Physical AI, where artificial intelligence meets the real world through embodied robotics and humanoid systems.
      </>
    ),
    link: '/docs/chapter-01-intro',
  },
  {
    title: 'ROS 2 Fundamentals',
    icon: '⚙️',
    chapter: 'Chapter 2',
    description: (
      <>
        Master Robot Operating System 2 with hands-on tutorials covering nodes, topics, services, and advanced control systems.
      </>
    ),
    link: '/docs/chapter-02-ros2',
  },
  {
    title: 'Gazebo Simulation',
    icon: '🌍',
    chapter: 'Chapter 3',
    description: (
      <>
        Build and test robots in virtual environments with Gazebo, learning URDF modeling, physics simulation, and sensor integration.
      </>
    ),
    link: '/docs/chapter-03-gazebo',
  },
  {
    title: 'NVIDIA Isaac Sim',
    icon: '🚀',
    chapter: 'Chapter 4',
    description: (
      <>
        Leverage GPU-accelerated robotics simulation with Isaac Sim for photorealistic environments and AI-powered robot training.
      </>
    ),
    link: '/docs/chapter-04-isaac',
  },
  {
    title: 'Vision-Language-Action Models',
    icon: '🧠',
    chapter: 'Chapter 5',
    description: (
      <>
        Explore cutting-edge VLA foundation models like RT-2 and OpenVLA that combine vision, language, and robotic manipulation.
      </>
    ),
    link: '/docs/chapter-05-vla',
  },
  {
    title: 'Humanoid Hardware Integration',
    icon: '🦾',
    chapter: 'Chapter 6',
    description: (
      <>
        Learn to integrate real humanoid robotics hardware, including actuators, sensors, embedded systems, and control interfaces.
      </>
    ),
    link: '/docs/chapter-06-hardware',
  },
];

function Feature({title, icon, description, link, chapter}: FeatureItem) {
  return (
    <div className={clsx('col col--4', styles.featureCol)}>
      <Link to={link} className={styles.featureCard}>
        <div className={styles.featureIcon}>{icon}</div>
        <div className={styles.featureChapter}>{chapter}</div>
        <Heading as="h3" className={styles.featureTitle}>{title}</Heading>
        <p className={styles.featureDescription}>{description}</p>
        <div className={styles.featureArrow}>
          →
        </div>
      </Link>
    </div>
  );
}

export default function HomepageFeatures(): ReactNode {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className={styles.featuresHeader}>
          <h2 className={styles.featuresHeading}>Comprehensive Learning Path</h2>
          <p className={styles.featuresSubheading}>
            Six expertly crafted chapters covering the entire Physical AI and Humanoid Robotics stack
          </p>
        </div>
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
