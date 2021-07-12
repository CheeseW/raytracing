#pragma once

#include "hittable.h"

#include <vector>
#include <memory>

namespace rayUtilities {
	class HittableList : public Hittable {
	public:
		__device__ HittableList(): objects(nullptr), size(0) {}
		__device__ HittableList(Hittable** obj, const int n) : objects(obj), size(n) {}

		__device__ virtual bool hit(const Ray& r, const float tMin, const float tMax, HitRecord& rec) const override;
		// __device__ int getSize() { return size; } const

	private:
		Hittable** objects;
		int size;
	};

	__device__ bool HittableList::hit(const Ray& r, const float tMin, const float tMax, HitRecord& rec) const
	{
		HitRecord temp;
		bool hit = false;
		auto closest = tMax;

		for (int i = 0; i < size; i++) {
			if (objects[i]->hit(r, tMin, closest, temp)) {
				hit = true;
				closest = temp.t;
				rec = temp;
			}
		}
		return hit;
	}
}