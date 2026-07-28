import { describe, expect, it } from "vitest";
import { placeholder } from "../src/index.js";

describe("smoke", () => {
  it("imports the app package", () => {
    expect(placeholder()).toBe("tryage-app");
  });
});
