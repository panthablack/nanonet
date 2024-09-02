import { createRouter, createWebHistory } from 'vue-router'
import DashboardView from '@/gui/views/DashboardView.vue'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/',
      name: 'dashboard',
      component: DashboardView,
    },
    {
      path: '/models',
      name: 'models',
      component: () => import('@/gui/views/models/ModelsView.vue'),
    },
    {
      path: '/models/train',
      name: 'model-train-index',
      component: () => import('@/gui/views/models/ModelTrainingIndexView.vue'),
    },
    {
      path: '/models/train/:id',
      name: 'model-train',
      component: () => import('@/gui/views/models/ModelTrainingView.vue'),
    },
    {
      path: '/models/test',
      name: 'model-test-index',
      component: () => import('@/gui/views/models/ModelTestingIndexView.vue'),
    },
    {
      path: '/models/test/:id',
      name: 'model-test',
      component: () => import('@/gui/views/models/ModelTestingView.vue'),
    },
    {
      path: '/models/:id',
      name: 'model',
      component: () => import('@/gui/views/models/ModelView.vue'),
    },
  ],
})

export default router
