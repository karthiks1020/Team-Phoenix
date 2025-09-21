import React, { createContext, useEffect, useMemo, useState } from 'react';

export const ThemeLangContext = createContext({
  theme: 'light',
  language: 'english',
  setTheme: () => {},
  setLanguage: () => {},
});

const applyTheme = (theme) => {
  const root = document.documentElement;
  if (theme === 'auto') {
    const prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
    root.setAttribute('data-theme', prefersDark ? 'dark' : 'light');
  } else {
    root.setAttribute('data-theme', theme);
  }
};

const applyLanguage = (language) => {
  const root = document.documentElement;
  root.setAttribute('lang', language === 'hindi' ? 'hi' : 'en');
  root.setAttribute('data-language', language);
};

export const ThemeLangProvider = ({ children }) => {
  const savedAccount = (() => {
    try {
      const raw = localStorage.getItem('accountSettings');
      return raw ? JSON.parse(raw) : null;
    } catch { return null; }
  })();

  const [theme, setTheme] = useState(savedAccount?.theme || 'light');
  const [language, setLanguage] = useState(savedAccount?.language || 'english');

  // React to changes
  useEffect(() => {
    applyTheme(theme);
    const existing = savedAccount || {};
    localStorage.setItem('accountSettings', JSON.stringify({ ...existing, theme }));
  }, [theme]);

  useEffect(() => {
    applyLanguage(language);
    const existing = savedAccount || {};
    localStorage.setItem('accountSettings', JSON.stringify({ ...existing, language }));
  }, [language]);

  const value = useMemo(() => ({ theme, language, setTheme, setLanguage }), [theme, language]);

  return (
    <ThemeLangContext.Provider value={value}>
      {children}
    </ThemeLangContext.Provider>
  );
};
