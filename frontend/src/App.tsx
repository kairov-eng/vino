import { BrowserRouter, Outlet, Route, Routes } from 'react-router-dom'
import { AgeGate } from './components/AgeGate'
import { Header } from './components/Header'
import { SiteAccessGate } from './components/SiteAccessGate'
import { HomePage } from './pages/HomePage'
import { InfoStubPage } from './pages/InfoStubPage'
import { NotFoundPage } from './pages/NotFoundPage'
import { ScanHistoryPage } from './pages/ScanHistoryPage'
import { ScanHistoryReportPage } from './pages/ScanHistoryReportPage'
import { WineDetailPage } from './pages/WineDetailPage'
import { WinesPage } from './pages/WinesPage'

function AppLayout() {
  return (
    <div className="app-shell">
      <Header />
      <Outlet />
      <AgeGate />
    </div>
  )
}

export default function App() {
  return (
    <SiteAccessGate>
      <BrowserRouter>
        <Routes>
          <Route element={<AppLayout />}>
            <Route path="/" element={<HomePage />} />
            <Route path="/wines" element={<WinesPage />} />
            <Route path="/wines/:slug" element={<WineDetailPage />} />
            <Route path="/history" element={<ScanHistoryPage />} />
            <Route
              path="/history/report/:reportKey"
              element={<ScanHistoryReportPage />}
            />
            <Route
              path="/cookies"
              element={<InfoStubPage title="cookies" />}
            />
            <Route
              path="/recommendations"
              element={<InfoStubPage title="рекомендательных технологий" />}
            />
          </Route>
          <Route path="*" element={<NotFoundPage />} />
        </Routes>
      </BrowserRouter>
    </SiteAccessGate>
  )
}
