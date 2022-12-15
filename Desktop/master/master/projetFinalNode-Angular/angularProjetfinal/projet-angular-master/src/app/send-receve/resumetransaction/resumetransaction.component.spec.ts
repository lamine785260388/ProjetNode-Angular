import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ResumetransactionComponent } from './resumetransaction.component';

describe('ResumetransactionComponent', () => {
  let component: ResumetransactionComponent;
  let fixture: ComponentFixture<ResumetransactionComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ ResumetransactionComponent ]
    })
    .compileComponents();

    fixture = TestBed.createComponent(ResumetransactionComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
