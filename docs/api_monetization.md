# Monetizacion de la API por llamada

Tres enfoques para cobrar por uso del endpoint de clasificacion de toxicidad en Google Cloud Platform, ordenados de menor a mayor complejidad y capacidad.

---

## 1. API Gateway + API Keys (ligero, sin monetizacion nativa)

API Gateway es un proxy gestionado que se configura con un OpenAPI spec. Autentica con API keys, aplica rate limiting y enruta a Cloud Run. No incluye facturacion nativa, pero provee los primitivos para construir un sistema de cobro propio.

### Arquitectura

```
Cliente ──▶ API Gateway ──▶ Cloud Run
               │
               ├── API key validation
               ├── Rate limiting (quota)
               └── Logging por key (para medir consumo)
```

### Configuracion paso a paso

#### 1.1. Crear el OpenAPI spec

```yaml
# openapi.yaml
swagger: "2.0"
info:
  title: "Toxic Comment Classifier API"
  version: "1.0.0"
host: "toxic-classifier-gateway.uc.gateway.cloud.google"
basePath: "/"
schemes:
  - "https"

x-google-backend:
  address: "https://toxic-comment-classifier-943214853579.us-central1.run.app"
  protocol: "h2"

securityDefinitions:
  api_key:
    type: "apiKey"
    name: "key"
    in: "query"

security:
  - api_key: []

paths:
  /health:
    get:
      operationId: "health"
      summary: "Health check"
      responses:
        200:
          description: "OK"
  /predict:
    post:
      operationId: "predict"
      summary: "Classify toxicity"
      consumes:
        - "application/json"
      produces:
        - "application/json"
      responses:
        200:
          description: "Prediction result"

x-google-management:
  metrics:
    - name: "predict-requests"
      displayName: "Prediction Requests"
      valueType: INT64
      metricKind: DELTA
  quota:
    limits:
      - name: "predict-requests-per-minute"
        metric: "predict-requests"
        unit: "1/min/{project}"
        values:
          STANDARD: 100
          PREMIUM: 1000
```

#### 1.2. Desplegar API Gateway

```bash
# Crear la API
gcloud api-gateway apis create toxic-classifier-api \
  --project=mlops-toxic-classifier

# Crear un API config desde el OpenAPI spec
gcloud api-gateway api-configs create toxic-classifier-config-v1 \
  --api=toxic-classifier-api \
  --openapi-spec=openapi.yaml \
  --project=mlops-toxic-classifier

# Crear el gateway
gcloud api-gateway gateways create toxic-classifier-gateway \
  --api=toxic-classifier-api \
  --api-config=toxic-classifier-config-v1 \
  --location=us-central1 \
  --project=mlops-toxic-classifier
```

#### 1.3. Crear API keys por cliente

```bash
# Habilitar la API para usar credentials
gcloud services enable apikeys.googleapis.com --project=mlops-toxic-classifier

# Crear una API key
gcloud alpha services api-keys create \
  --display-name="client-alpha" \
  --project=mlops-toxic-classifier

# Restringir la key al API Gateway
gcloud alpha services api-keys update KEY_ID \
  --api-target=api=toxic-classifier-api \
  --project=mlops-toxic-classifier
```

#### 1.4. Medir consumo y facturar manualmente

API Gateway registra cada llamada en Cloud Logging con la API key. Para facturar, se consultan los logs y se calcula el consumo por cliente.

```bash
# Contar llamadas por API key en un periodo
gcloud logging read \
  'resource.type="api_gateway" AND resource.labels.gateway_name="toxic-classifier-gateway"' \
  --project=mlops-toxic-classifier \
  --format=json \
  --limit=10000 | \
python3 -c "
import sys, json, collections
data = json.load(sys.stdin)
counts = collections.Counter()
for entry in data:
    key = entry.get('jsonPayload', {}).get('api_key', 'unknown')
    counts[key] += 1
for key, count in counts.most_common():
    print(f'{key[:20]}...: {count} calls')
"
```

Alternativa mas robusta: exportar logs a BigQuery y calcular facturacion con SQL.

```bash
# Crear sink de Cloud Logging a BigQuery
gcloud logging sinks create api-usage-sink \
  bigquery.googleapis.com/projects/mlops-toxic-classifier/datasets/api_usage \
  --log-filter='resource.type="api_gateway"' \
  --project=mlops-toxic-classifier
```

```sql
-- Consulta de facturacion mensual por cliente
SELECT
  json_extract(jsonPayload, '$.api_key') AS api_key,
  COUNT(*) AS total_calls,
  COUNT(*) * 0.001 AS charge_usd
FROM
  `mlops-toxic-classifier.api_usage.api_gateway_*`
WHERE
  _TABLE_SUFFIX BETWEEN '20260601' AND '20260630'
GROUP BY
  api_key
ORDER BY
  charge_usd DESC;
```

### Costo de API Gateway

| Concepto | Precio |
|---|---|
| Primeras 2M llamadas/mes | Gratis |
| Llamadas adicionales | $3.00 por millon |

### Limitaciones

- No tiene portal de desarrolladores.
- No tiene facturacion nativa. Hay que construir el pipeline de cobro (logs -> BigQuery -> calculo -> Stripe/invoice).
- Solo soporta API keys y service accounts. No OAuth.
- Rate limiting por proyecto, no por developer individual (a menos que se creen proyectos separados).

### Cuando usarlo

La API es un detalle interno de una aplicacion propia. Se necesita autenticacion y rate limiting basico, pero la facturacion se resuelve fuera de GCP o de forma manual.

---

## 2. Apigee (monetizacion nativa, portal de desarrolladores)

Apigee es la plataforma de gestion de APIs de Google Cloud. Incluye monetizacion nativa con rate plans, developer portals, analiticas, y politicas programables. Es la solucion si la API es un producto con consumidores externos que pagan.

### Arquitectura

```
Cliente ──▶ Apigee API Proxy ──▶ Cloud Run
               │
               ├── API key / OAuth
               ├── Rate limiting por developer
               ├── MonetizationLimitsCheck policy
               ├── Developer portal (self-service)
               ├── Rate plans (pay-per-call, tiered, freemium, subscription)
               └── Analytics + billing reports
```

### Configuracion paso a paso

#### 2.1. Crear organizacion de Apigee

```bash
# Habilitar Apigee
gcloud services enable apigee.googleapis.com --project=mlops-toxic-classifier

# Crear organizacion de Apigee (pay-as-you-go)
gcloud apigee organizations create mlops-toxic-classifier \
  --analytics-region=us-central1 \
  --project=mlops-toxic-classifier
```

#### 2.2. Crear API proxy hacia Cloud Run

En la UI de Apigee o via API:

1. Crear un API proxy con el backend `https://toxic-comment-classifier-943214853579.us-central1.run.app`.
2. Agregar politica de verificacion de API key.
3. Agregar politica de rate limiting.
4. Desplegar el proxy al entorno de Apigee.

#### 2.3. Crear API Product

```bash
# API Product = la oferta que los developers compran
curl -X POST \
  "https://apigee.googleapis.com/v1/organizations/mlops-toxic-classifier/apiproducts" \
  -H "Authorization: Bearer $(gcloud auth print-access-token)" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "toxic-classifier-premium",
    "displayName": "Toxic Comment Classifier - Premium",
    "description": "Multi-label toxicity classification API with TF-IDF + nomic-embed",
    "approvalType": "AUTO",
    "attributes": [
      {"name": "access", "value": "public"}
    ],
    "proxies": ["toxic-classifier-proxy"],
    "environments": ["prod"]
  }'
```

#### 2.4. Habilitar monetizacion

```bash
# Habilitar monetization en la organizacion
curl -X POST \
  "https://apigee.googleapis.com/v1/organizations/mlops-toxic-classifier:setAddons" \
  -H "Authorization: Bearer $(gcloud auth print-access-token)" \
  -H "Content-Type: application/json" \
  -d '{
    "addonsConfig": {
      "monetizationConfig": {
        "enabled": true
      }
    }
  }'
```

#### 2.5. Definir rate plans

Apigee soporta cuatro modelos de pricing:

**Pay-per-call** (cobrar por cada llamada):

```bash
curl -X POST \
  "https://apigee.googleapis.com/v1/organizations/mlops-toxic-classifier/apiproducts/toxic-classifier-premium/rateplans" \
  -H "Authorization: Bearer $(gcloud auth print-access-token)" \
  -H "Content-Type: application/json" \
  -d '{
    "displayName": "Pay Per Call",
    "apiproduct": "toxic-classifier-premium",
    "description": "$0.001 per prediction request",
    "state": "PUBLISHED",
    "startTime": "1767225600000",
    "currencyCode": "USD",
    "consumptionPricingType": "FIXED_PER_UNIT",
    "consumptionPricingRates": [
      {
        "fee": {
          "currencyCode": "USD",
          "units": "0",
          "nanos": 1000000
        }
      }
    ],
    "billingPeriod": "MONTHLY"
  }'
```

**Tiered pricing** (descuento por volumen):

```bash
curl -X POST \
  "https://apigee.googleapis.com/v1/organizations/mlops-toxic-classifier/apiproducts/toxic-classifier-premium/rateplans" \
  -H "Authorization: Bearer $(gcloud auth print-access-token)" \
  -H "Content-Type: application/json" \
  -d '{
    "displayName": "Volume Tiered Pricing",
    "apiproduct": "toxic-classifier-premium",
    "description": "Discounted rates at higher volumes",
    "state": "PUBLISHED",
    "startTime": "1767225600000",
    "currencyCode": "USD",
    "consumptionPricingType": "BANDED",
    "consumptionPricingRates": [
      {
        "start": "0",
        "end": "10000",
        "fee": {"currencyCode": "USD", "units": "0", "nanos": 2000000}
      },
      {
        "start": "10001",
        "end": "100000",
        "fee": {"currencyCode": "USD", "units": "0", "nanos": 1000000}
      },
      {
        "start": "100001",
        "fee": {"currencyCode": "USD", "units": "0", "nanos": 500000}
      }
    ],
    "billingPeriod": "MONTHLY"
  }'
```

Esto produce tres tramos:

| Llamadas | Precio por llamada |
|---|---|
| 0 - 10,000 | $0.002 |
| 10,001 - 100,000 | $0.001 |
| 100,001+ | $0.0005 |

**Freemium** (uso gratis hasta un limite, luego se cobra):

```bash
curl -X POST \
  "https://apigee.googleapis.com/v1/organizations/mlops-toxic-classifier/apiproducts/toxic-classifier-premium/rateplans" \
  -H "Authorization: Bearer $(gcloud auth print-access-token)" \
  -H "Content-Type: application/json" \
  -d '{
    "displayName": "Freemium Plan",
    "apiproduct": "toxic-classifier-premium",
    "description": "1,000 free calls per month, then $0.001 per call",
    "state": "PUBLISHED",
    "startTime": "1767225600000",
    "currencyCode": "USD",
    "consumptionPricingType": "BANDED",
    "consumptionPricingRates": [
      {
        "start": "0",
        "end": "1000",
        "fee": {"currencyCode": "USD", "units": "0", "nanos": 0}
      },
      {
        "start": "1001",
        "fee": {"currencyCode": "USD", "units": "0", "nanos": 1000000}
      }
    ],
    "billingPeriod": "MONTHLY"
  }'
```

**Suscripcion fija** (precio plano mensual sin importar uso):

```bash
curl -X POST \
  "https://apigee.googleapis.com/v1/organizations/mlops-toxic-classifier/apiproducts/toxic-classifier-premium/rateplans" \
  -H "Authorization: Bearer $(gcloud auth print-access-token)" \
  -H "Content-Type: application/json" \
  -d '{
    "displayName": "Enterprise Subscription",
    "apiproduct": "toxic-classifier-premium",
    "description": "$99/month unlimited access",
    "state": "PUBLISHED",
    "startTime": "1767225600000",
    "currencyCode": "USD",
    "fixedRecurringFee": {
      "currencyCode": "USD",
      "units": "99"
    },
    "billingPeriod": "MONTHLY"
  }'
```

#### 2.6. Registrar developers y suscribirlos

```bash
# Registrar un developer
curl -X POST \
  "https://apigee.googleapis.com/v1/organizations/mlops-toxic-classifier/developers" \
  -H "Authorization: Bearer $(gcloud auth print-access-token)" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "client@example.com",
    "firstName": "Client",
    "lastName": "Alpha",
    "status": "ACTIVE"
  }'

# Suscribir el developer al rate plan
curl -X POST \
  "https://apigee.googleapis.com/v1/organizations/mlops-toxic-classifier/developers/client@example.com/subscriptions" \
  -H "Authorization: Bearer $(gcloud auth print-access-token)" \
  -H "Content-Type: application/json" \
  -d '{
    "apiproduct": "toxic-classifier-premium",
    "startTime": "1769904000000"
  }'
```

#### 2.7. Enforcer limites de monetizacion en el proxy

Agregar la politica `MonetizationLimitsCheck` al proxy para bloquear accesos de developers sin suscripcion o con saldo insuficiente:

```xml
<!-- apiproxy/policies/CheckMonetizationLimits.xml -->
<MonetizationLimitsCheck continueOnError="false" enabled="true" name="CheckMonetizationLimits">
    <DisplayName>Check Monetization Limits</DisplayName>
    <IgnoreUnresolvedVariables>true</IgnoreUnresolvedVariables>
    <FaultResponse>
        <Set>
            <Payload contentType="application/json">
                {"error":"API product subscription is missing or prepaid balance is insufficient"}
            </Payload>
            <StatusCode>403</StatusCode>
        </Set>
    </FaultResponse>
</MonetizationLimitsCheck>
```

#### 2.8. Consultar reportes de facturacion

```bash
# Balance de un developer prepaid
curl "https://apigee.googleapis.com/v1/organizations/mlops-toxic-classifier/developers/client@example.com/balance" \
  -H "Authorization: Bearer $(gcloud auth print-access-token)"

# Reporte de uso por developer y producto
curl --get \
  "https://apigee.googleapis.com/v1/organizations/mlops-toxic-classifier/environments/prod/stats/developer_email,api_product" \
  -H "Authorization: Bearer $(gcloud auth print-access-token)" \
  --data-urlencode "select=sum(x_apigee_mintng_rate)" \
  --data-urlencode "timeRange=06/01/2026 00:00~06/30/2026 23:59" \
  --data-urlencode "timeUnit=month"
```

Alternativa: exportar analytics a BigQuery para facturacion detallada.

```sql
-- Facturacion mensual por developer
SELECT
  developer_email,
  api_product,
  COUNT(*) AS total_calls,
  CASE
    WHEN COUNT(*) <= 1000 THEN 0
    WHEN COUNT(*) <= 10000 THEN (COUNT(*) - 1000) * 0.002
    WHEN COUNT(*) <= 100000 THEN (9000 * 0.002) + ((COUNT(*) - 10000) * 0.001)
    ELSE (9000 * 0.002) + (90000 * 0.001) + ((COUNT(*) - 100000) * 0.0005)
  END AS estimated_cost_usd
FROM
  `mlops-toxic-classifier.apigee_analytics.api_*`
WHERE
  _TABLE_SUFFIX BETWEEN '20260601' AND '20260630'
GROUP BY
  developer_email, api_product
ORDER BY
  estimated_cost_usd DESC;
```

### Costo de Apigee

| Concepto | Precio |
|---|---|
| Enviroment fee (pay-as-you-go) | Desde $365/mes por ambiente |
| Llamadas | ~$20 por millon (pay-as-you-go) |
| Subscription | Desde ~$500/mes |
| Monetizacion | Incluida en subscription, add-on en pay-as-you-go |

### Cuando usarlo

La API es un producto con consumidores externos que pagan. Se necesita portal de desarrolladores, rate plans con multiples modelos de pricing, tracking automatico de uso por developer, y facturacion integrada.

---

## 3. Middleware custom + Stripe (maxima flexibilidad, sin vendor lock-in)

Si Apigee es demasiado costoso o se necesita integracion directa con un procesador de pagos (Stripe), se puede construir un middleware en Cloud Run que actue como gateway propio.

### Arquitectura

```
Cliente ──▶ Cloud Run (middleware/gateway) ──▶ Cloud Run (predictor)
               │
               ├── API key validation (desde Firestore/Datastore)
               ├── Rate limiting (Redis o en-memoria)
               ├── Usage counting (Firestore/BigQuery)
               ├── Stripe Customer/Subs API
               └── Webhook de Stripe para eventos de pago
```

### Componentes

#### 3.1. Middleware FastAPI

```python
# src/serving/gateway.py
from fastapi import FastAPI, Request, HTTPException, Depends
from google.cloud import firestore
import stripe, time, os

app = FastAPI(title="Toxic Classifier API Gateway")

db = firestore.Client(project="mlops-toxic-classifier")
stripe.api_key = os.environ["STRIPE_SECRET_KEY"]
PREDICTOR_URL = "https://toxic-comment-classifier-943214853579.us-central1.run.app"

RATE_LIMITS = {
    "free": 100,       # llamadas por dia
    "basic": 10000,
    "pro": 100000,
    "enterprise": -1,  # sin limite
}

PLAN_PRICES = {
    "free": 0,
    "basic": 29,       # USD/mes
    "pro": 99,
    "enterprise": 299,
}

async def verify_api_key(request: Request):
    key = request.headers.get("X-API-Key", "")
    if not key:
        raise HTTPException(401, "Missing X-API-Key header")

    doc = db.collection("api_keys").document(key).get()
    if not doc.exists:
        raise HTTPException(401, "Invalid API key")

    data = doc.to_dict()
    plan = data.get("plan", "free")
    customer_id = data.get("stripe_customer_id")

    # Rate limiting
    today = time.strftime("%Y-%m-%d")
    counter_doc = db.collection("usage").document(f"{key}_{today}").get()
    count = counter_doc.to_dict().get("count", 0) if counter_doc.exists else 0

    limit = RATE_LIMITS[plan]
    if limit > 0 and count >= limit:
        raise HTTPException(429, f"Rate limit exceeded for {plan} plan. Upgrade at /billing")

    # Incrementar contador
    db.collection("usage").document(f"{key}_{today}").set({"count": count + 1})

    return {"key": key, "plan": plan, "customer_id": customer_id}


@app.post("/predict")
async def predict(request: Request, auth=Depends(verify_api_key)):
    import httpx
    body = await request.json()
    resp = await httpx.post(f"{PREDICTOR_URL}/predict", json=body, timeout=60)
    return resp.json()


@app.get("/usage")
async def get_usage(auth=Depends(verify_api_key)):
    key = auth["key"]
    usages = []
    for doc in db.collection("usage").where("count", ">", 0).stream():
        if doc.id.startswith(key):
            usages.append({"date": doc.id.split("_")[1], "count": doc.to_dict()["count"]})
    return {"plan": auth["plan"], "usage": usages}
```

#### 3.2. Stripe integration

```python
# Crear un customer en Stripe cuando se registra un developer
@app.post("/register")
async def register(email: str, plan: str = "free"):
    customer = stripe.Customer.create(email=email)
    db.collection("api_keys").document(generate_key()).set({
        "email": email,
        "plan": plan,
        "stripe_customer_id": customer.id,
        "created": firestore.SERVER_TIMESTAMP,
    })

    if plan != "free":
        # Crear suscripcion en Stripe
        price_id = STRIPE_PRICE_IDS[plan]
        stripe.Subscription.create(
            customer=customer.id,
            items=[{"price": price_id}],
        )

    return {"message": "Registered", "plan": plan}


# Webhook de Stripe para actualizar plan cuando se paga
@app.post("/stripe/webhook")
async def stripe_webhook(request: Request):
    payload = await request.body()
    event = stripe.Webhook.construct_event(
        payload, request.headers.get("Stripe-Signature"), STRIPE_WEBHOOK_SECRET
    )
    if event["type"] == "customer.subscription.updated":
        sub = event["data"]["object"]
        customer_id = sub["customer"]
        new_plan = get_plan_from_price(sub["items"]["data"][0]["price"]["id"])
        # Actualizar plan en Firestore
        for doc in db.collection("api_keys").where("stripe_customer_id", "==", customer_id).stream():
            doc.reference.update({"plan": new_plan})
    return {"status": "ok"}
```

#### 3.3. Desplegar el middleware

```bash
gcloud run deploy toxic-classifier-gateway \
  --image=us-central1-docker.pkg.dev/mlops-toxic-classifier/mlops-containers/toxic-gateway:latest \
  --region=us-central1 \
  --set-env-vars="STRIPE_SECRET_KEY=sk_...,PREDICTOR_URL=https://..." \
  --allow-unauthenticated \
  --memory=512Mi
```

### Costo del middleware custom

| Componente | Precio estimado |
|---|---|
| Cloud Run (gateway) | ~$5/mes para trafico bajo |
| Firestore | Gratuito hasta 50k ops/dia |
| Stripe | 2.9% + $0.30 por transaccion |
| BigQuery (analytics) | $6.25/TB escaneado |

### Cuando usarlo

Se necesita integracion directa con Stripe u otro procesador de pagos. Se quiere control total sobre el flujo de autenticacion, rate limiting y facturacion. No se quiere depender de Apigee. La escala es baja a media (<1M llamadas/mes).

---

## Comparacion de los tres enfoques

| Criterio | API Gateway | Apigee | Middleware + Stripe |
|---|---|---|---|
| **Complejidad de setup** | Baja | Alta | Media |
| **Portal de desarrolladores** | No | Si (integrado) | No (construir propio) |
| **Facturacion nativa** | No | Si (rate plans, prepaid/postpaid) | No (via Stripe) |
| **Modelos de pricing** | Manual (logs -> calculo) | Per-call, tiered, freemium, subscription | Cualquiera (codigo custom) |
| **Autenticacion** | API key, service account | API key, OAuth | Cualquiera (custom) |
| **Rate limiting** | Por proyecto | Por developer | Por developer (custom) |
| **Analytics** | Cloud Logging | Apigee Analytics + BigQuery export | Firestore + BigQuery |
| **Costo base** | ~$0 (2M free) | ~$365/mes + $20/M calls | ~$5/mes + Stripe fees |
| **Costo a 1M calls/mes** | $3 | $365 + $20 = ~$385 | $5 + Stripe |
| **Costo a 10M calls/mes** | $24 | $365 + $200 = ~$565 | $5 + Stripe |
| **Vendor lock-in** | GCP | GCP (Apigee) | Bajo (Stripe es portable) |
| **Tiempo de implementacion** | 1-2 dias | 1-2 semanas | 3-5 dias |

---

## Recomendacion para este proyecto

La eleccion depende de la fase del producto:

**Fase 1 (MVP, <10k calls/mes):** API Gateway. Autenticacion con API keys, rate limiting basico, logs a BigQuery para medir consumo. Cobro manual (invoice mensual) o integracion externa con Stripe. Costo cercano a cero.

**Fase 2 (producto con clientes pagando, <1M calls/mes):** Apigee pay-as-you-go. Portal de desarrolladores donde los clientes se registran y compran rate plans. Monetizacion nativa con prepaid/postpaid. Facturacion automatica. Costo ~$385/mes a 1M calls.

**Fase 3 (escala alta, integracion de pagos custom):** Middleware propio + Stripe. Control total del flujo de cobro, sin dependencia de Apigee. Costo ~$5/mes en infraestructura + comisiones de Stripe. Requiere mas desarrollo pero es mas portable y economico a escala.

---

## Precios sugeridos para la API de toxicidad

Basado en el costo de infraestructura y el valor del servicio:

| Plan | Precio | Incluye |
|---|---|---|
| Free | $0/mes | 1,000 llamadas/mes, 10 textos por request |
| Basic | $29/mes | 10,000 llamadas/mes, 50 textos por request |
| Pro | $99/mes | 100,000 llamadas/mes, 128 textos por request, SLA 99.5% |
| Enterprise | $299/mes | Llamadas ilimitadas, 128 textos por request, SLA 99.9%, soporte dedicado |

**Pay-per-call alternativo:** $0.001 por prediccion (1 millon de llamadas = $1,000/mes).

El costo marginal de una prediccion es aproximadamente $0.0001 (Cloud Run ~450ms a $0.000024/vCPU-sec + llamada a Synthetic API gratuita). Un margen de 10x sobre costo marginal ($0.001 por call) deja margen para cubrir desarrollo, monitoreo y fraude.
