/** Re-export settings admin auth context (site itself is public). */
export {
  ADMIN_PASSWORD_COOKIE,
  AdminPasswordForm,
  SiteAuthProvider as SiteAccessGate,
  readAdminPasswordCookie,
  useSiteAuth,
} from '../auth/SiteAuthContext'
