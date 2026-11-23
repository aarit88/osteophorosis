export const SUPPORTED_UPLOAD_MEDIA_TYPES = new Set([
  'image/png', 'image/jpeg', 'image/webp', 'image/tiff', 'image/bmp',
]);

export function resolveCorsOrigins(raw: string | undefined, production: boolean): string[] {
  const origins = (raw || (production ? '' : 'http://localhost:3000,http://localhost:5173,http://127.0.0.1:3000,http://127.0.0.1:5173'))
    .split(',').map((value) => value.trim()).filter(Boolean);
  if (production && (!origins.length || origins.includes('*'))) {
    throw new Error('Production requires explicit OSTEO_CORS_ORIGINS without wildcard origins.');
  }
  if (production && origins.some((origin) => {
    try {
      const parsed = new URL(origin);
      const local = parsed.hostname === 'localhost' || parsed.hostname === '127.0.0.1';
      return parsed.origin !== origin || (parsed.protocol !== 'https:' && !local);
    } catch { return true; }
  })) {
    throw new Error('Production CORS origins must be exact HTTPS origins (localhost may use HTTP).');
  }
  return origins;
}

export function isStrongProductionSecret(secret: string | undefined): boolean {
  return Boolean(secret && secret.length >= 32);
}

export function isSupportedUploadMediaType(mediaType: string): boolean {
  return SUPPORTED_UPLOAD_MEDIA_TYPES.has(mediaType.toLowerCase());
}
