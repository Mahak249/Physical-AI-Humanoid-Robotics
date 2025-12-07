import React from 'react';
import Layout from '@theme/Layout';
import UserProfile from '../components/UserProfile';

function SettingsPage() {
  return (
    <Layout title="Settings">
      <div className="container">
        <h1>User Settings</h1>
        <UserProfile />
      </div>
    </Layout>
  );
}

export default SettingsPage;
