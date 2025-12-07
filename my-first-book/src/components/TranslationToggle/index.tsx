import React, { useState } from 'react';

function TranslationToggle() {
  const [language, setLanguage] = useState('en');

  const handleToggle = () => {
    const newLanguage = language === 'en' ? 'ur' : 'en';
    setLanguage(newLanguage);
    // Here you would typically trigger a re-fetch of the content
    // with the new language.
  };

  return (
    <div>
      <button onClick={handleToggle}>
        {language === 'en' ? 'Switch to Urdu' : 'Switch to English'}
      </button>
    </div>
  );
}

export default TranslationToggle;
