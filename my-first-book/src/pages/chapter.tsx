import React from 'react';
import Layout from '@theme/Layout';
import TranslationToggle from '../components/TranslationToggle';

function ChapterPage() {
  return (
    <Layout title="Chapter">
      <div className="container">
        <h1>Chapter Title</h1>
        <TranslationToggle />
        <p>Chapter content goes here.</p>
      </div>
    </Layout>
  );
}

export default ChapterPage;
