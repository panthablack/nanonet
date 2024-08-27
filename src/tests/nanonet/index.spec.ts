import { describe, it, expect, expectTypeOf } from 'vitest'
import type { NanoNet } from '@/types/NanoNet'

import { createNanonet } from '@/nanonet'

describe('createNanonet', () => {
  it('should be a function', () => {
    expect(typeof createNanonet).toBe('function')
  })

  it('should return an instance of NanoNet', () => {
    expectTypeOf(createNanonet()).toMatchTypeOf<NanoNet>()
  })
})
