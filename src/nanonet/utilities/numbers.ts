export const getRandomFloat = (x: number = 0, y: number = 1): number => {
  // if numbers equal, return that number
  if (x === y) return x

  // set min to lowest number and max to highest
  const min = Math.min(x, y)
  const max = Math.max(x, y)

  // generate sense of randomness
  const randomness = Math.random()

  // get distance
  const distance = Math.abs(max - min)

  // get random value size
  const travel = distance * randomness

  // return distance travelled from the min
  return min + travel
}
