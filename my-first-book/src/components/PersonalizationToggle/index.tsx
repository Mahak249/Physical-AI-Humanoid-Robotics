import React, { useState } from 'react';

function PersonalizationToggle() {
  const [isPersonalized, setIsPersonalized] = useState(false);

  const handleToggle = () => {
    setIsPersonalized(!isPersonalized);
    // Here you would typically trigger a re-fetch of the content
    // with the personalization option.
  };

  return (
    <div>
      <label>
        <input type="checkbox" checked={isPersonalized} onChange={handleToggle} />
        Enable Personalized Content
      </label>
    </div>
  );
}

export default PersonalizationToggle;
