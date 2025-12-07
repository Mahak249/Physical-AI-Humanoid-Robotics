import type {ReactNode} from 'react';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import HomepageFeatures from '@site/src/components/HomepageFeatures';
import HeroSection from '@site/src/components/HeroSection';
import Chatbot from '@site/src/components/Chatbot';

export default function Home(): ReactNode {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`${siteConfig.title}`}
      description="Master Physical AI and Humanoid Robotics with hands-on learning">
      <HeroSection />
      <main>
        <HomepageFeatures />
      </main>
      <Chatbot />
    </Layout>
  );
}
