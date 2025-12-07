import React, { useState, useEffect } from 'react';

function UserProfile() {
  const [background, setBackground] = useState('');
  const [userId, setUserId] = useState(1); // Hardcoded for now

  useEffect(() => {
    // Fetch user profile
    fetch(`/api/users/${userId}/profile`)
      .then(response => response.json())
      .then(data => {
        if (data) {
          setBackground(data.background);
        }
      });
  }, [userId]);

  const handleSubmit = (event) => {
    event.preventDefault();
    // Create or update user profile
    fetch(`/api/users/${userId}/profile`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ background }),
    });
  };

  return (
    <form onSubmit={handleSubmit}>
      <label>
        Background:
        <textarea value={background} onChange={(e) => setBackground(e.target.value)} />
      </label>
      <button type="submit">Save</button>
    </form>
  );
}

export default UserProfile;
