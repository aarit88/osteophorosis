FROM node:20-bookworm-slim AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
ARG VITE_API_BASE_URL=
ENV VITE_API_BASE_URL=${VITE_API_BASE_URL}
RUN npm run build && npm prune --omit=dev

FROM node:20-bookworm-slim AS runtime
ENV NODE_ENV=production \
    HOST=0.0.0.0 \
    PORT=3000 \
    HOME=/tmp
WORKDIR /app
COPY --from=build --chown=node:node /app/package.json /app/package-lock.json ./
COPY --from=build --chown=node:node /app/node_modules ./node_modules
COPY --from=build --chown=node:node /app/server.ts /app/security.ts /app/security-policy.ts /app/research-api.ts /app/clinical-readiness.ts /app/dataset-governance.ts ./
COPY --from=build --chown=node:node /app/mock-server/simulation.js ./mock-server/simulation.js
COPY --from=build --chown=node:node /app/openapi.yaml ./openapi.yaml
COPY --from=build --chown=node:node /app/dist ./dist
COPY --from=build --chown=node:node /app/ml_artifacts ./ml_artifacts
RUN mkdir -p /var/lib/osteoguard/node && chown -R node:node /var/lib/osteoguard
USER node
EXPOSE 3000
HEALTHCHECK --interval=20s --timeout=4s --start-period=15s --retries=3 \
  CMD node -e "fetch('http://127.0.0.1:3000/readyz').then(r=>process.exit(r.ok?0:1)).catch(()=>process.exit(1))"
CMD ["npm", "start"]
