import { createRouter, createWebHistory } from 'vue-router'
import DashboardView from '@/dev/views/DashboardView.vue'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/',
      name: 'dashboard',
      component: DashboardView,
    },
    {
      path: '/models/test',
      name: 'model-test',
      component: () => import('@/dev/views/ModelTestingView.vue'),
    },
  ],
})

export default router
