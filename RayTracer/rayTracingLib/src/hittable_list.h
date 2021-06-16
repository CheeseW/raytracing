#pragma once

#include "hittable.h"

#include <vector>
#include <memory>

namespace rayUtilities {
	class HittableList : public Hittable {
	public:
		HittableList() {}
		HittableList(std::shared_ptr<Hittable> obj) { add(obj); }

		void clear() { objects.clear(); }
		void add(std::shared_ptr<Hittable> obj) { objects.push_back(obj); };

		virtual bool hit(const Ray& r, const double tMin, const double tMax, HitRecord& rec) const override;

	private:
		std::vector<std::shared_ptr<Hittable>> objects;
	};

	
}