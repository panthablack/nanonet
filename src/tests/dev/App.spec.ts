import { describe, it, expect } from 'vitest'

import { shallowMount } from '@vue/test-utils'
import App from '@/dev/App.vue'

describe('App', () => {
  it('renders properly', () => {
    const wrapper = shallowMount(App)
    expect(wrapper.html()).toContain('appContainer')
  })
})
