import '@/gui/assets/css/index.css'

import { createApp } from 'vue'
import { createPinia } from 'pinia'

import App from '@/gui/App.vue'
import router from '@/gui/router'

const app = createApp(App)

app.use(createPinia())
app.use(router)

app.mount('#app')
