import type { ComponentType, LazyExoticComponent } from 'react';
import { lazy } from 'react';
import {
  Activity,
  Compass,
  Cpu,
  Database,
  Eye,
  FileCheck2,
  Info,
  Layers,
  Lock,
  Scale,
  Sliders,
} from 'lucide-react';
import { Home } from '../pages/Home';

type PageComponent = ComponentType | LazyExoticComponent<ComponentType>;

const Results = lazy(() => import('../pages/Results').then((module) => ({ default: module.Results })));
const About = lazy(() => import('../pages/About').then((module) => ({ default: module.About })));
const AuditPage = lazy(() => import('../pages/AuditPage').then((module) => ({ default: module.AuditPage })));
const ResearchWorkbenchPage = lazy(() => import('../pages/ResearchWorkbenchPage').then((module) => ({ default: module.ResearchWorkbenchPage })));
const ResearchTracksPage = lazy(() => import('../pages/ResearchTracksPage').then((module) => ({ default: module.ResearchTracksPage })));
const RadiomicsPage = lazy(() => import('../pages/RadiomicsPage').then((module) => ({ default: module.RadiomicsPage })));
const BiomechanicsPage = lazy(() => import('../pages/BiomechanicsPage').then((module) => ({ default: module.BiomechanicsPage })));
const ExplainabilityPage = lazy(() => import('../pages/ExplainabilityPage').then((module) => ({ default: module.ExplainabilityPage })));
const UncertaintyPage = lazy(() => import('../pages/UncertaintyPage').then((module) => ({ default: module.UncertaintyPage })));
const EvidenceRegistryPage = lazy(() => import('../pages/EvidenceRegistryPage').then((module) => ({ default: module.EvidenceRegistryPage })));
const GovernancePage = lazy(() => import('../pages/GovernancePage').then((module) => ({ default: module.GovernancePage })));
const ArchitecturePage = lazy(() => import('../pages/ArchitecturePage').then((module) => ({ default: module.ArchitecturePage })));

export interface ApplicationRoute {
  path: string;
  title: string;
  description: string;
  component: PageComponent;
  icon: typeof Activity;
  navigation: 'primary' | 'research' | 'support' | 'hidden';
}

/** Single source of truth for implemented pages and visible navigation. */
export const APPLICATION_ROUTES: readonly ApplicationRoute[] = [
  { path: '/', title: 'Home', description: 'OsteoGuard research overview', component: Home, icon: Activity, navigation: 'primary' },
  { path: '/workbench', title: 'Workbench', description: 'Research demonstrations and tools', component: ResearchWorkbenchPage, icon: Cpu, navigation: 'primary' },
  { path: '/tracks', title: 'Research tracks', description: 'Track status and research boundaries', component: ResearchTracksPage, icon: Layers, navigation: 'primary' },
  { path: '/radiomics', title: 'Radiomics', description: 'Radiomics research tools', component: RadiomicsPage, icon: Sliders, navigation: 'research' },
  { path: '/biomechanics', title: 'Biomechanics', description: 'Biomechanics and Mirels research tools', component: BiomechanicsPage, icon: Compass, navigation: 'research' },
  { path: '/explainability', title: 'Explainability', description: 'Explainability research tools', component: ExplainabilityPage, icon: Eye, navigation: 'research' },
  { path: '/uncertainty', title: 'Uncertainty', description: 'Uncertainty and abstention research', component: UncertaintyPage, icon: Activity, navigation: 'research' },
  { path: '/governance', title: 'Governance', description: 'Dataset governance, onboarding and preparation', component: GovernancePage, icon: Database, navigation: 'primary' },
  { path: '/evidence', title: 'Evidence registry', description: 'Research evidence registry', component: EvidenceRegistryPage, icon: FileCheck2, navigation: 'primary' },
  { path: '/architecture', title: 'Architecture', description: 'Software architecture and Track D readiness gate', component: ArchitecturePage, icon: Lock, navigation: 'primary' },
  { path: '/audit', title: 'Audit', description: 'Research audit information', component: AuditPage, icon: Scale, navigation: 'support' },
  { path: '/about', title: 'About', description: 'Project scope and methodology', component: About, icon: Info, navigation: 'support' },
  { path: '/results', title: 'Results', description: 'Analysis result details', component: Results, icon: Activity, navigation: 'hidden' },
] as const;

export const ROUTE_BY_PATH = new Map(APPLICATION_ROUTES.map((route) => [route.path, route]));
