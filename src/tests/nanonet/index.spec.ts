import { describe, it, expect, expectTypeOf } from 'vitest'
import type { NanoNetModel } from '@/types/NanoNet'

import { createModel } from '@/nanonet'

describe('createModel', () => {
  it('should be a function', () => {
    expect(typeof createModel).toBe('function')
  })

  it('should return an instance of NanoNetModel', () => {
    expectTypeOf(createModel()).toMatchTypeOf<NanoNetModel>()
  })
})
