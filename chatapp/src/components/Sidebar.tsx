import React from 'react';
import { PresetQuery } from '../types/index';

export interface SidebarProps {
  onSelectPreset: (query: string) => void;
}

export function Sidebar({ onSelectPreset }: SidebarProps): React.JSX.Element {
  const presets: PresetQuery[] = [
    {
      id: 'eligibility',
      icon: '🩸',
      title: 'Donor Eligibility',
      desc: 'Evaluate donor criteria using classification tool',
      query: 'Check donor eligibility for age 30, weight 70kg, last donation 2024-01-01'
    },
    {
      id: 'stockout',
      icon: '🏥',
      title: 'Stockout Hazard Prediction',
      desc: 'Run hazard rate survival model for hospital inventory',
      query: 'Predict blood supply stockout risk for regional hospital'
    },
    {
      id: 'inventory',
      icon: '📦',
      title: 'Hospital Inventory Query',
      desc: 'Inspect current stock levels via database tool',
      query: 'Show central hospital inventory for blood type O+'
    }
  ];

  return (
    <aside className="sidebar">
      <div>
        <div className="sidebar-section-title">
          <i className="fa-solid fa-bolt"></i> Sample Queries
        </div>

        {presets.map(preset => (
          <div
            key={preset.id}
            className="preset-card"
            onClick={() => onSelectPreset(preset.query)}
          >
            <div className="preset-title">
              <span>{preset.icon}</span> {preset.title}
            </div>
            <div className="preset-desc">{preset.desc}</div>
          </div>
        ))}
      </div>
    </aside>
  );
}
