import { formatTime, midiName, pct, externalSearchLinks } from "@/lib/utils";
import { translate } from "@/lib/i18n";

describe("utils", () => {
  it("formats values", () => {
    expect(formatTime(75.4)).toBe("1:15");
    expect(formatTime(null)).toBe("–");
    expect(pct(0.256, 1)).toBe("25.6%");
    expect(midiName(60)).toBe("C4");
    expect(midiName(69)).toBe("A4");
  });
  it("builds search links without leaking anything but title and artist", () => {
    const links = externalSearchLinks("Paper Boats", "Me & You");
    expect(links.every((l) => l.url.includes("Paper%20Boats%20Me%20%26%20You"))).toBe(true);
  });
  it("translates with interpolation and falls back to English", () => {
    expect(translate("es", "nav.library")).toBe("Biblioteca");
    expect(translate("en", "rec.min", { s: 5 })).toContain("5s");
  });
});
